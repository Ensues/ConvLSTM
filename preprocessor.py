def preprocess_and_cache_videos(video_folder, cache_folder, transforms, num_frames=30):
    """
    Pre-extract and cache video frames for faster training.
    
    Args:
        video_folder: Path to video files
        cache_folder: Path to save cached frames
        transforms: Torchvision transforms to apply
        num_frames: Number of frames per video (default: 30)
    """
    # Create cache directory if it doesn't exist
    os.makedirs(cache_folder, exist_ok=True)
    
    # Get all video files
    video_files = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]
    
    print(f"Starting frame extraction for {len(video_files)} videos...")
    print(f"Cache directory: {cache_folder}")
    print(f"This may take a few minutes but only needs to be done once.\n")
    
    start_time = time.time()
    cached_count = 0
    skipped_count = 0
    
    for video_name in tqdm(video_files, desc="Preprocessing videos"):
        video_path = os.path.join(video_folder, video_name)
        cache_path = os.path.join(cache_folder, video_name.replace('.mp4', '.npy'))
        
        # Skip if already cached
        if os.path.exists(cache_path):
            skipped_count += 1
            continue
        
        # Extract frames from video
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        for _ in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                # Pad with black frame if video is shorter
                frame_tensor = torch.zeros((3, HEIGHT, WIDTH))
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if transforms:
                    frame = Image.fromarray(frame)
                    frame_tensor = transforms(frame)
                else:
                    frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            
            frames.append(frame_tensor)
        
        cap.release()
        
        # Stack and save as numpy array
        video_tensor = torch.stack(frames, dim=0)  # [30, 3, 128, 128]
        np.save(cache_path, video_tensor.numpy())
        
        cached_count += 1
        del frames, video_tensor
    
    elapsed_time = time.time() - start_time
    
    print(f"\n✓ Preprocessing completed!")
    print(f"  - Newly cached: {cached_count} videos")
    print(f"  - Already cached: {skipped_count} videos")
    print(f"  - Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"  - Cache size: ~{(cached_count + skipped_count) * 2:.0f} MB")
    print(f"\n✓ Ready for training! The dataset will now load from cache.")