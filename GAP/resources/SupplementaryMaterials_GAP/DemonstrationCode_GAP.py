if __name__ == "__main__":
    # Initialize arguments and optimizer
    args, op = init_args()

    # Create output directory with descriptive name
    output_dir = os.path.join(
        args.output_dir, 
        f"{args.seed}-{args.viewpoint_mode[0]}{args.num_viewpoints}-"
        f"{args.update_mode[0]}{args.update_steps}-{args.new_strength}-"
        f"{args.update_strength}-{args.view_threshold}"
    )
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"=> OUTPUT_DIR: {output_dir}")

    # ==================== MODEL INITIALIZATION ====================
    # Extract model name and prepare checkpoints
    name = args.obj_file[:-4]
    ckpt_path = f''
    
    # Process input mesh: sample points and convert to PLY format
    sample_points(os.path.join(args.input_dir, args.obj_file), output_dir)
    conver_ply(output_dir)
    
    # Initialize Gaussian model
    gaussians = GaussianModel()
    gaussians.load_ply(os.path.join(output_dir, 'point_cloud.ply'))
    
    # Setup UDF network with hyperparameters
    udf_network = CAPUDFNetwork(
        d_out=1,
        d_in=3,
        d_hidden=256,
        n_layers=8,
        skip_in=[4],
        multires=0,
        bias=0.5,
        scale=1.0,
        geometric_init=True,
        weight_norm=True
    ).cuda()

    # Load pre-trained weights and set to evaluation mode
    udf_network.load_state_dict(
        torch.load(ckpt_path, map_location=torch.device('cuda'))["udf_network_fine"]
    )
    udf_network = udf_network.eval()
    
    # Initialize update tracking tensor
    update_tensor = torch.zeros([gaussians._xyz.shape[0]], dtype=torch.bool).cuda()
    
    # ==================== VIEWPOINT SETUP ====================
    # Initialize viewpoints for generation and refinement
    principle_directions = None
    (
        dist_list, 
        elev_list, 
        azim_list, 
        sector_list,
        view_punishments, 
        length
    ) = init_viewpoints(
        args.viewpoint_mode, 
        args.num_viewpoints, 
        args.dist, 
        args.elev, 
        principle_directions, 
        use_principle=True, 
        use_shapenet=args.use_shapenet,
        use_objaverse=args.use_objaverse
    )
    
    # Save configuration for reproducibility
    save_args(args, output_dir)

    # Initialize depth2image model for generation
    controlnet, ddim_sampler = get_controlnet_depth()

    # ==================== CAMERA PREPARATION ====================
    # Get principal viewpoints
    NUM_PRINCIPLE = length
    pre_dist_list = dist_list[:NUM_PRINCIPLE]
    pre_elev_list = elev_list[:NUM_PRINCIPLE]
    pre_azim_list = azim_list[:NUM_PRINCIPLE]
    pre_sector_list = sector_list[:NUM_PRINCIPLE]
    pre_view_punishments = view_punishments[:NUM_PRINCIPLE]

    # Initialize camera transformation matrices
    R_list = []
    T_list = []
    R_raw_list = []
    T_raw_list = []
    
    for view_idx in range(NUM_PRINCIPLE):
        # Get camera parameters for this viewpoint
        dist = pre_dist_list[view_idx]
        elev = pre_elev_list[view_idx]
        azim = pre_azim_list[view_idx]
        sector = pre_sector_list[view_idx]
        
        # Set up camera and convert to appropriate format
        camera = init_camera(dist, elev, azim, args.image_size, DEVICE)
        R, T = convert_camera_from_pytorch3d_to_gs(camera, args.image_size, args.image_size)
        
        # Store camera matrices
        R_list.append(R)
        T_list.append(T.reshape(3))

    # Create scene with cameras and Gaussian model
    scene = Scene(R_list, T_list, gaussians, image_size=args.image_size)
    gaussians.training_setup(op, scene.cameras_extent)

    # Build similarity cache for all viewpoints
    similarity_view_cache = build_similarity_gaussian_cache_for_all_views_gaussian(
        pre_dist_list, pre_elev_list, pre_azim_list,
        args.image_size, args.image_size * args.render_simple_factor, 
        args.uv_size, args.fragment_k,
        gaussians, scene,
        DEVICE, udf_network
    )
   
    # Initialize containers for tracking updates
    update_views = []
    masks = []
    visibilitys = []

    # ==================== Gaussian GENERATION PROCESS ====================
    print("=> start generating gaussian...")
    start_time = time.time()
    new_gaussian = None
    
    for view_idx in range(NUM_PRINCIPLE):
        print(f"=> processing view {view_idx}...")

        # Get camera parameters for current view
        dist = pre_dist_list[view_idx]
        elev = pre_elev_list[view_idx]
        azim = pre_azim_list[view_idx]
        sector = pre_sector_list[view_idx]
        
        # Create prompt for text-to-image generation
        prompt = f" the {sector} view of {args.prompt}, {sector} view"
        
        # ===== STEP 1: Render View and Create Masks =====
        (
            view_score,
            cameras,
            init_image, normal_map, depth_map, 
            init_images_tensor, normal_maps_tensor, depth_maps_tensor, similarity_tensor, 
            keep_mask_image, update_mask_image, generate_mask_image, 
            keep_mask_tensor, update_mask_tensor, generate_mask_tensor, 
            all_mask_tensor, quad_mask_tensor, visibility_filter
        ) = render_one_view_and_build_masks_gaussian(
            dist, elev, azim, 
            view_idx, view_idx, view_punishments,
            similarity_view_cache,
            args.image_size, args.fragment_k,
            init_image_dir, mask_image_dir, normal_map_dir, 
            depth_map_dir, similarity_map_dir, gaussian_dir,
            DEVICE, 
            scene,
            udf_network, new_gaussian, R_raw_list, T_raw_list, gaussians,
            save_intermediate=True, 
            smooth_mask=args.smooth_mask, 
            view_threshold=args.view_threshold
        )
   
        # Track visibility for this view
        visibilitys.append(visibility_filter)

        # ===== STEP 2: Generate Missing Regions =====
        print(f"=> generating image for prompt: {prompt}...")
        
        # Handle no-repaint option for views after the first
        if args.no_repaint and view_idx != 0:
            actual_generate_mask_image = Image.fromarray(
                (np.ones_like(np.array(generate_mask_image)) * 255.).astype(np.uint8)
            )
        else:
            actual_generate_mask_image = generate_mask_image

        print(f"=> generate for view {view_idx}")

        # Only process if enough area needs updating (>1%)
        update_ratio = update_mask_tensor.sum() / all_mask_tensor.sum() if all_mask_tensor.sum() > 0 else 0
        if update_mask_tensor.sum() > 0 and update_ratio >= 0.01:
            # Set refinement strength
            refine_strength = 0.6
            
            # Apply depth-conditioned inpainting
            generate_image, generate_image_before, generate_image_after, generate_image_tensor = apply_controlnet_depth_intex(
                intex,
                init_image.convert('RGBA'), 
                prompt,
                generate_mask_tensor, 
                update_mask_tensor, 
                keep_mask_tensor,
                init_images_tensor, 
                depth_maps_tensor.permute(1, 2, 0).repeat(1, 1, 3).cpu().numpy(),
                args.a_prompt, 
                args.n_prompt, 
                refine_strength
            )
        
            # Save intermediate generation results
            generate_image.save(os.path.join(inpainted_image_dir, f"{view_idx}.png"))
            generate_image_before.save(os.path.join(inpainted_image_dir, f"{view_idx}_before.png"))
            generate_image_after.save(os.path.join(inpainted_image_dir, f"{view_idx}_after.png"))
            
            # ===== STEP 3: Update Gaussian Model =====
            update_view, mask, update, new_gaussian = opt_gaussian_from_one_view_generate_and_update(
                gaussians, 
                scene, 
                view_idx, 
                generate_image_tensor, 
                generate_mask_tensor, 
                op, 
                init_images_tensor.squeeze(0), 
                keep_mask_tensor, 
                update_mask_tensor, 
                masks[view_idx], 
                visibility_filter, 
                dist, elev, azim, sector, 
                DEVICE, 
                udf_network, 
                None, 
                gaussian_dir
            )
    
    # ==================== FINAL 3D PROCESSING ====================
    print("#" * 40)
    print("start 3D Inpainting")
    print("#" * 40)
    
    # Update point colors and save final model
    gaussians = update_colored_points(gaussians, gaussian_dir)
    gaussians.save_ply(os.path.join(gaussian_dir, "final.ply"))
    
    # Report total processing time
    total_time = time.time() - start_time
    print(f"=> total generate time: {total_time:.2f} seconds")