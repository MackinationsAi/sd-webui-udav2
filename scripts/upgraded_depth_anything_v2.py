import numpy as np
from PIL import Image
from safetensors.torch import load_file
from depth_anything_v2.dpt import DepthAnythingV2
import cv2
from tqdm import tqdm
import matplotlib
from modules import devices, script_callbacks, shared, util, paths_internal, modelloader
from modules.scripts import basedir
import os
import gradio as gr
import tempfile
import glob
import time

# sd-webui-udav2 extension code by: MackinationsAi
# original underlying code by: DepthAnything


shared.options_templates.update(shared.options_section(('saving-paths', 'Paths for saving'), {
    'outdir_udav2': shared.OptionInfo(util.truncate_path(os.path.join(paths_internal.default_output_dir, 'depths', 'vis_image_depths')), 'Output directory for UDAV2 images', component_args=shared.hide_dirs),
}))


extension_dir = basedir()
checkpoints_dir = os.path.join(extension_dir, 'checkpoints')
DEVICE = devices.get_device_for('udav2')

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}

checkpoints = {
    'vits': ("depth_anything_v2_vits.safetensors", 'https://huggingface.co/MackinationsAi/Depth-Anything-V2_Safetensors/resolve/main/depth_anything_v2_vits.safetensors', 'a7c1a8c8cdd7885fb8391069cd1eee789126c8d896f7de6750499b1097f817ea'),
    'vitb': ("depth_anything_v2_vitb.safetensors", 'https://huggingface.co/MackinationsAi/Depth-Anything-V2_Safetensors/resolve/main/depth_anything_v2_vitb.safetensors', '386758cbd2a2cac62ca62286d3ba810734561b3097d86a585dd3dac357153941'),
    'vitl': ("depth_anything_v2_vitl.safetensors", 'https://huggingface.co/MackinationsAi/Depth-Anything-V2_Safetensors/resolve/main/depth_anything_v2_vitl.safetensors', 'f075a9099f94bae54a5bfe21a1423346429309bae40abb85b9935985b1f35a09'),
}


def load_model(encoder):
    checkpoint_filename, checkpoint_url, sha256 = checkpoints.get(encoder, None)
    checkpoint_path = modelloader.load_file_from_url(checkpoint_url, model_dir=checkpoints_dir, file_name=checkpoint_filename, hash_prefix=sha256)

    model = DepthAnythingV2(**model_configs[encoder])
    state_dict = load_file(checkpoint_path)
    model.load_state_dict(state_dict)
    model = model.to(DEVICE).eval()
    return model


def predict_depth(model, image):
    return model.infer_image(image)


def save_image_with_structure(image, prefix, suffix, output_base_path):
    filename = get_next_filename(output_base_path, prefix, suffix, '.png')
    image.save(filename)
    return filename


def save_colourized_image_with_structure(image, prefix, suffix, colour_map_method, output_base_path):
    filename = get_next_filename(output_base_path, prefix, f'_{colour_map_method}{suffix}', '.png')
    image.save(filename)
    return filename


def save_video_with_structure(base_path, base_filename, suffix):
    filename = get_next_filename(base_path, f'_{base_filename}', suffix, '.mp4')
    return filename


def get_next_filename(base_path, prefix, suffix, extension):
    i = 1
    while os.path.exists(f"{base_path}/{i:04d}{prefix}{suffix}{extension}"):
        i += 1
    return f"{base_path}/{i:04d}{prefix}{suffix}{extension}"


full_colour_map_methods = [
    'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'GnBu', 'GnBu_r', 'Grays', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 
    'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_grey', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gist_yerg', 
    'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'grey', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r',  'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r', 'All'
]

colour_map_methods = [
    'Spectral', 'terrain', 'viridis', 'plasma', 'inferno', 'magma', 'cividis',
    'twilight', 'rainbow', 'gist_rainbow', 'gist_ncar', 'gist_earth', 'turbo',
    'jet', 'afmhot', 'copper', 'seismic', 'hsv', 'brg', 'All'
]


def process_image(image, colour_map_method, encoder, selection):
    model = load_model(encoder)
    original_image = image.copy()
    h, w = image.shape[:2]
    depth = predict_depth(model, image[:, :, ::-1])
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255
    depth = depth.astype(np.uint8)
    grey_depth = Image.fromarray(depth)

    date_str = time.strftime("%Y-%m-%d")
    output_base_path = os.path.join(shared.opts.outdir_udav2, date_str)
    os.makedirs(output_base_path, exist_ok=True)

    grey_depth_filename = save_image_with_structure(grey_depth, '', '_greyscale_depth_map', output_base_path)

    if colour_map_method == 'All':
        colourized_filenames = []
        methods = full_colour_map_methods[:-1] if selection == "Full" else colour_map_methods[:-1]  # Exclude 'All' from the methods list
        for method in methods:
            coloured_depth = (matplotlib.colormaps.get_cmap(method)(depth)[:, :, :3] * 255).astype(np.uint8)
            coloured_depth_image = Image.fromarray(coloured_depth)
            depth_filename = save_colourized_image_with_structure(coloured_depth_image, '', '_coloured_depth_map', method, output_base_path)
            colourized_filenames.append(depth_filename)
        return colourized_filenames, grey_depth_filename
    else:
        coloured_depth = (matplotlib.colormaps.get_cmap(colour_map_method)(depth)[:, :, :3] * 255).astype(np.uint8)
        coloured_depth_image = Image.fromarray(coloured_depth)
        depth_filename = save_colourized_image_with_structure(coloured_depth_image, '', '_coloured_depth_map', colour_map_method, output_base_path)
        return [depth_filename], grey_depth_filename


def process_video(video_paths, output_path, input_size, encoder, colour_map_method, greyscale):
    model = load_model(encoder)

    date_str = time.strftime("%Y-%m-%d")
    if not output_path:
        output_base_path = os.path.join(shared.opts.outdir_udav2, date_str)
    else:
        output_base_path = output_path
    os.makedirs(output_base_path, exist_ok=True)

    margin_width = 13

    for k, video in enumerate(video_paths):
        filename = video.name if isinstance(video, tempfile._TemporaryFileWrapper) else video
        print(f'Progress {k+1}/{len(video_paths)}: {filename}')

        raw_video = cv2.VideoCapture(filename)

        frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = int(raw_video.get(cv2.CAP_PROP_FPS))
        output_width = frame_width * 2 + margin_width
        base_filename = os.path.splitext(os.path.basename(filename))[0]
        combined_output_path = save_video_with_structure(output_base_path, base_filename, '_combined')
        greyscale_depth_output_path = save_video_with_structure(output_base_path, base_filename, '_depth_greyscale')
        colourized_depth_output_path = save_video_with_structure(output_base_path, base_filename, '_depth_colourized')
        combined_out = cv2.VideoWriter(combined_output_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (output_width, frame_height))
        greyscale_depth_out = cv2.VideoWriter(greyscale_depth_output_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (frame_width, frame_height))
        colourized_depth_out = cv2.VideoWriter(colourized_depth_output_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (frame_width, frame_height))
        frame_count = int(raw_video.get(cv2.CAP_PROP_FRAME_COUNT))

        for frame_index in tqdm(range(frame_count), desc="Processing frames", unit="frame"):
            ret, raw_frame = raw_video.read()
            if not ret:
                break

            depth = model.infer_image(raw_frame, input_size)
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255
            depth = depth.astype(np.uint8)
            depth_grey = np.repeat(depth[..., np.newaxis], 3, axis=-1)

            if raw_frame.shape != depth_grey.shape:
                print(f"Skipping frame {frame_index} due to shape mismatch: {raw_frame.shape} vs {depth_grey.shape}")
                continue

            greyscale_depth_out.write(depth_grey)
            cmap = matplotlib.colormaps.get_cmap(colour_map_method)
            depth_colour = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

            if raw_frame.shape != depth_colour.shape:
                print(f"Skipping frame {frame_index} due to shape mismatch: {raw_frame.shape} vs {depth_colour.shape}")
                continue

            colourized_depth_out.write(depth_colour)
            split_region = np.ones((frame_height, margin_width, 3), dtype=np.uint8) * 255

            try:
                combined_frame = cv2.hconcat([raw_frame, split_region, depth_colour])
            except cv2.error as e:
                print(f"Error concatenating frame {frame_index}: {e}")
                continue

            combined_out.write(combined_frame)

        raw_video.release()
        combined_out.release()
        greyscale_depth_out.release()
        colourized_depth_out.release()

        return [], combined_output_path

    return [], None


def sort_by_creation_time(file_list):
    return sorted(file_list, key=os.path.getctime)


css = """
#img-display-container {
    max-height: 300vh;
}
#img-display-input {
    max-height: 160vh;
}
#img-display-output {
    max-height: 160vh;
}
#download {
    height: 26px;
}
button[aria-label="Clear"] {
    display: none !important;
}
.gallery-item img {
    height: 150px; /* Adjust this value as needed */
    object-fit: cover; /* Ensure the image covers the specified height */
}
"""


def on_ui_tabs():
    title = """
    # <span style="font-size: 1em; color: #FF5733;">[Upgraded Depth Anything V2](https://github.com/MackinationsAi/Upgraded-Depth-Anything-V2) 🚀 </span>
    """

    with gr.Blocks(css=css) as UDAV2:
        gr.Markdown(title)

        encoders = ['vits', 'vitb', 'vitl']

        colour_map_selections = ["Default", "Full"]

        def get_colour_map_methods(selection):
            return full_colour_map_methods if selection == "Full" else colour_map_methods

        with gr.Tab("Single Image Processing"):
            with gr.Row():
                model_encoder_image_single = gr.Dropdown(label="Select Model Encoder:", choices=encoders, value='vitl')
            with gr.Row():
                input_image = gr.Image(label="Input Image", type='filepath', elem_id='img-display-input', height=794, display_fn=lambda x: x)
                depth_image_slider = gr.Image(label="Colourized Depth Map View", elem_id='img-display-output', height=794, display_fn=lambda x: x)
                grey_depth_file_single = gr.Image(label="Greyscale Depth Map View", elem_id='img-display-output', height=794, display_fn=lambda x: x)
            with gr.Row():
                with gr.Column(scale=8):
                    colour_map_dropdown_single = gr.Dropdown(label="Select Colour Map Method:", choices=colour_map_methods, value='Spectral')
                with gr.Column(scale=1):
                    colour_map_selection_single = gr.Dropdown(label="Colour Map Method Selection:", choices=colour_map_selections, value='Default')
            submit_single = gr.Button(value="Compute Depth for Single Image", variant="primary", height=26)

            def on_submit_single(image_path, colour_map_method, encoder, selection):
                image = np.array(Image.open(image_path))
                colourized_filenames, grey_depth_filename = process_image(image, colour_map_method, encoder, selection)
                first_colourized_image = Image.open(colourized_filenames[0])
                first_colourized_image_np = np.array(first_colourized_image)
                return first_colourized_image_np, grey_depth_filename

            colour_map_selection_single.change(fn=lambda selection: gr.update(choices=get_colour_map_methods(selection)), inputs=colour_map_selection_single, outputs=colour_map_dropdown_single)

        with gr.Tab("Batch Image Processing"):
            with gr.Row():
                model_encoder_image_batch = gr.Dropdown(label="Select Model Encoder:", choices=encoders, value='vitl')
            with gr.Row():
                input_images = gr.Files(label="Upload Images", type="file", elem_id="img-display-input")
            with gr.Row():
                with gr.Column(scale=8):
                    colour_map_dropdown_batch = gr.Dropdown(label="Select Colour Map Method:", choices=colour_map_methods, value='Spectral')
                with gr.Column(scale=1):
                    colour_map_selection_batch = gr.Dropdown(label="Colour Map Method Selection:", choices=colour_map_selections, value='Default')
            submit_batch = gr.Button(value="Compute Depth for Batch", variant="primary")
            output_message = gr.Textbox(label="Output", lines=1, interactive=False)

            def on_batch_submit(files, colour_map_method, encoder, selection):
                results = []
                for file in files:
                    file_path = file.name if isinstance(file, tempfile._TemporaryFileWrapper) else file
                    image = np.array(Image.open(file_path))
                    colourized_filenames, grey_filename = process_image(image, colour_map_method, encoder, selection)
                    results.append(f"Processed {os.path.basename(file_path)}: {', '.join(colourized_filenames)}, {grey_filename}")
                return "\n".join(results)

            colour_map_selection_batch.change(fn=lambda selection: gr.update(choices=get_colour_map_methods(selection)), inputs=colour_map_selection_batch, outputs=colour_map_dropdown_batch)

        with gr.Tab("Single Video Processing"):
            with gr.Row():
                model_encoder_video_single = gr.Dropdown(label="Select Model Encoder:", choices=encoders, value='vitl')
            with gr.Row():
                input_video_single = gr.Video(label="Input Video", elem_id='img-display-input', height=650)
            with gr.Row():
                with gr.Column(scale=8):
                    colour_map_dropdown_video_single = gr.Dropdown(label="Select Colour Map Method:", choices=colour_map_methods, value='Spectral')
                with gr.Column(scale=1):
                    colour_map_selection_video_single = gr.Dropdown(label="Colour Map Method Selection:", choices=colour_map_selections, value='Default')
            with gr.Accordion(open=False, label="Advanced Options:"):
                with gr.Row():
                    with gr.Column(scale=1):
                        output_dir_single = gr.Textbox(label="Output Directory:", value='outputs/depths/vis_vid_depths')
                    with gr.Column(scale=1):
                        input_size_single = gr.Slider(label="Input Size:", minimum=256, maximum=1024, step=1, value=1024)
            greyscale_single = gr.State(value=False)
            submit_video_single = gr.Button(value="Compute Depth for Single Video", variant="primary")
            output_message_video_single = gr.Textbox(label="Outputs", lines=1, interactive=False)

            def on_submit_video_single(video, colour_map_method, output_dir, input_size, greyscale, encoder):
                _, combined_output_path = process_video([video], output_dir, input_size, encoder, colour_map_method, greyscale)
                return combined_output_path

            colour_map_selection_video_single.change(fn=lambda selection: gr.update(choices=get_colour_map_methods(selection)), inputs=colour_map_selection_video_single, outputs=colour_map_dropdown_video_single)

        with gr.Tab("Batch Video Processing"):
            with gr.Row():
                model_encoder_video_batch = gr.Dropdown(label="Select Model Encoder:", choices=encoders, value='vitl')
            with gr.Row():
                input_videos = gr.Files(label="Input Videos", type="file", elem_id='img-display-input')
            with gr.Row():
                with gr.Column(scale=5):
                    colour_map_dropdown_video_batch = gr.Dropdown(label="Select Colour Map Method:", choices=colour_map_methods, value='Spectral')
                with gr.Column(scale=1):
                    colour_map_selection_video_batch = gr.Dropdown(label="Colour Map Method Selection:", choices=colour_map_selections, value='Default')
                with gr.Column(scale=1):
                    output_dir = gr.Textbox(label="Default Output Directory:", value='outputs/depths/vis_videos_depths/batch')
            submit_video_batch = gr.Button(value="Compute Depth for Video(s)", variant="primary")
            output_message_video = gr.Textbox(label="Outputs", lines=1, interactive=False)

            def on_submit_video_batch(videos, colour_map_method, output_dir, encoder):
                results = []
                for video in videos:
                    filename = video.name if isinstance(video, tempfile._TemporaryFileWrapper) else video
                    _, combined_output_path = process_video([filename], output_dir, 1024, encoder, colour_map_method, False)
                    result_message = f"Processed {os.path.basename(filename)}: Combined video: {combined_output_path}"
                    results.append(result_message)
                return "\n".join(results)

            colour_map_selection_video_batch.change(fn=lambda selection: gr.update(choices=get_colour_map_methods(selection)), inputs=colour_map_selection_video_batch, outputs=colour_map_dropdown_video_batch)

        with gr.Tab("Today's Depth Gallery"):
            gr.Markdown()

            date_str = time.strftime("%Y-%m-%d")
            gallery_path = os.path.join(shared.opts.outdir_udav2, date_str, '*')
            example_files = glob.glob(gallery_path)
            example_files = sort_by_creation_time(example_files)  # Sort files by creation time

            gallery = gr.Gallery(value=example_files, label="", show_label=False, elem_id="gallery", columns=[5], object_fit="contain", height="auto")

            def display_selected_image(img_paths):
                if img_paths and len(img_paths) > 0:
                    img_path = img_paths[0] if isinstance(img_paths[0], str) else img_paths[0]['name']
                    return Image.open(img_path)
                return None

            gallery.select(display_selected_image, inputs=[gallery])

        def update_gallery():
            date_str = time.strftime("%Y-%m-%d")
            gallery_path = os.path.join(shared.opts.outdir_udav2, date_str, '*')
            example_files = glob.glob(gallery_path)
            example_files = sort_by_creation_time(example_files)
            return example_files

        submit_single.click(on_submit_single, inputs=[input_image, colour_map_dropdown_single, model_encoder_image_single, colour_map_selection_single], outputs=[depth_image_slider, grey_depth_file_single]).then(update_gallery, outputs=[gallery])
        submit_batch.click(on_batch_submit, inputs=[input_images, colour_map_dropdown_batch, model_encoder_image_batch, colour_map_selection_batch], outputs=[output_message]).then(update_gallery, outputs=[gallery])
        submit_video_single.click(on_submit_video_single, inputs=[input_video_single, colour_map_dropdown_video_single, output_dir_single, input_size_single, greyscale_single, model_encoder_video_single], outputs=[output_message_video_single]).then(update_gallery, outputs=[gallery])
        submit_video_batch.click(on_submit_video_batch, inputs=[input_videos, colour_map_dropdown_video_batch, output_dir, model_encoder_video_batch], outputs=[output_message_video]).then(update_gallery, outputs=[gallery])

    return [(UDAV2, "UDAV2", "udav2")]


script_callbacks.on_ui_tabs(on_ui_tabs)
