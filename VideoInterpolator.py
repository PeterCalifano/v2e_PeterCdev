from v2ecore.slomo import SuperSloMo
import os 

slomo_model_checkpoint: str = ""
slowdown_factor: int = 1
auto_timestamp_resolution: bool = True
skip_video_output: bool = True
output_folder: str = "output_slomo"
video_path: str | None = None if skip_video_output else output_folder
vid_orig: str | None = None if skip_video_output else "vid_orig_path"
vid_slomo: str | None = None if skip_video_output else "vid_slomo_path"
output_preview: bool = False if skip_video_output else True
batch_size: int = 1

def DefineSloMo():

    slomo = SuperSloMo(
        model=slomo_model_checkpoint,
        auto_upsample=auto_timestamp_resolution,
        upsampling_factor=slowdown_factor,
        video_path=video_path,
        vid_orig=vid_orig,     
        vid_slomo=vid_slomo,   
        preview=output_preview,
        batch_size=batch_size)  
    
    return slomo


def InterpolateFrames(model, source_frames_dir:str, interp_frames_folder:str, frame_size: list | tuple):

    # Create the output directory if it doesn't exist
    os.makedirs(interp_frames_folder, exist_ok=True)

    print("Interpolating frames...")
    # Call SloMo to interpolate the frames and produce output video
    interpTimes, avgUpsamplingFactor = model.interpolate(source_frame_path=source_frames_dir,
                                                         output_folder=interp_frames_folder,
                                                         frame_size=frame_size)
    # Print the average upsampling factor
    print(f"Average upsampling factor: {avgUpsamplingFactor}")


def main():

    source_frames_dir = "source_frames_dir"
    interp_frames_folder = "interp_frames_folder"
    frame_size = (256, 256)  # Example frame size

    slomo_model = DefineSloMo()
    InterpolateFrames(slomo_model, source_frames_dir=source_frames_dir, interp_frames_folder=interp_frames_folder, frame_size=frame_size)

if __name__ == "__main__":
    main()

