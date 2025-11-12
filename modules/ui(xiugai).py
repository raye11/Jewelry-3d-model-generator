import datetime
import mimetypes
import os
import sys
from functools import reduce
import warnings
from contextlib import ExitStack

import gradio as gr
import gradio.utils
import numpy as np
from PIL import Image, PngImagePlugin  # noqa: F401
from modules.call_queue import wrap_gradio_gpu_call, wrap_queued_call, wrap_gradio_call, wrap_gradio_call_no_job # noqa: F401

from modules import gradio_extensons, sd_schedulers  # noqa: F401
from modules import sd_hijack, sd_models, script_callbacks, ui_extensions, deepbooru, extra_networks, ui_common, progress, ui_loadsave, shared_items, ui_settings, timer, sysinfo, ui_checkpoint_merger, scripts, sd_samplers, processing, ui_extra_networks, ui_toprow, launch_utils
from modules.ui_components import FormRow, FormGroup, ToolButton, FormHTML, InputAccordion, ResizeHandleRow
from modules.paths import script_path
from modules.ui_common import create_refresh_button
from modules.ui_gradio_extensions import reload_javascript

from modules.shared import opts, cmd_opts

import modules.infotext_utils as parameters_copypaste
import modules.hypernetworks.ui as hypernetworks_ui
import modules.textual_inversion.ui as textual_inversion_ui
import modules.textual_inversion.textual_inversion as textual_inversion
import modules.shared as shared
from modules import prompt_parser
from modules.sd_hijack import model_hijack
from modules.infotext_utils import image_from_url_text, PasteField
import tempfile

create_setting_component = ui_settings.create_setting_component

if shared.opts is not None:
    warnings.filterwarnings("default" if shared.opts.show_warnings else "ignore", category=UserWarning)
    warnings.filterwarnings("default" if shared.opts.show_gradio_deprecation_warnings else "ignore", category=gr.deprecation.GradioDeprecationWarning)

# this is a fix for Windows users. Without it, javascript files will be served with text/html content-type and the browser will not show any UI
mimetypes.init()
mimetypes.add_type('application/javascript', '.js')
mimetypes.add_type('application/javascript', '.mjs')

# Likewise, add explicit content-type header for certain missing image types
mimetypes.add_type('image/webp', '.webp')
mimetypes.add_type('image/avif', '.avif')

if not cmd_opts.share and not cmd_opts.listen:
    # fix gradio phoning home
    gradio.utils.version_check = lambda: None
    gradio.utils.get_local_ip_address = lambda: '127.0.0.1'

if cmd_opts.ngrok is not None:
    import modules.ngrok as ngrok
    print('ngrok authtoken detected, trying to connect...')
    ngrok.connect(
        cmd_opts.ngrok,
        cmd_opts.port if cmd_opts.port is not None else 7860,
        cmd_opts.ngrok_options
        )


def gr_show(visible=True):
    return {"visible": visible, "__type__": "update"}


sample_img2img = "assets/stable-samples/img2img/sketch-mountains-input.jpg"
sample_img2img = sample_img2img if os.path.exists(sample_img2img) else None

# Using constants for these since the variation selector isn't visible.
# Important that they exactly match script.js for tooltip to work.
random_symbol = '\U0001f3b2\ufe0f'  # 🎲️
reuse_symbol = '\u267b\ufe0f'  # ♻️
paste_symbol = '\u2199\ufe0f'  # ↙
refresh_symbol = '\U0001f504'  # 🔄
save_style_symbol = '\U0001f4be'  # 💾
apply_style_symbol = '\U0001f4cb'  # 📋
clear_prompt_symbol = '\U0001f5d1\ufe0f'  # 🗑️
extra_networks_symbol = '\U0001F3B4'  # 🎴
switch_values_symbol = '\U000021C5' # ⇅
restore_progress_symbol = '\U0001F300' # 🌀
detect_image_size_symbol = '\U0001F4D0'  # 📐


plaintext_to_html = ui_common.plaintext_to_html


def send_gradio_gallery_to_image(x):
    if len(x) == 0:
        return None
    return image_from_url_text(x[0])


def calc_resolution_hires(enable, width, height, hr_scale, hr_resize_x, hr_resize_y):
    if not enable:
        return ""

    p = processing.StableDiffusionProcessingTxt2Img(width=width, height=height, enable_hr=True, hr_scale=hr_scale, hr_resize_x=hr_resize_x, hr_resize_y=hr_resize_y)
    p.calculate_target_resolution()

    return f"from <span class='resolution'>{p.width}x{p.height}</span> to <span class='resolution'>{p.hr_resize_x or p.hr_upscale_to_x}x{p.hr_resize_y or p.hr_upscale_to_y}</span>"



def resize_from_to_html(width, height, scale_by):
    target_width = int(width * scale_by)
    target_height = int(height * scale_by)

    if not target_width or not target_height:
        return "no image selected"

    return f"resize: from <span class='resolution'>{width}x{height}</span> to <span class='resolution'>{target_width}x{target_height}</span>"


def process_interrogate(interrogation_function, mode, ii_input_dir, ii_output_dir, *ii_singles):
    if mode in {0, 1, 3, 4}:
        return [interrogation_function(ii_singles[mode]), None]
    elif mode == 2:
        return [interrogation_function(ii_singles[mode]["image"]), None]
    elif mode == 5:
        assert not shared.cmd_opts.hide_ui_dir_config, "Launched with --hide-ui-dir-config, batch img2img disabled"
        images = shared.listfiles(ii_input_dir)
        print(f"Will process {len(images)} images.")
        if ii_output_dir != "":
            os.makedirs(ii_output_dir, exist_ok=True)
        else:
            ii_output_dir = ii_input_dir

        for image in images:
            img = Image.open(image)
            filename = os.path.basename(image)
            left, _ = os.path.splitext(filename)
            print(interrogation_function(img), file=open(os.path.join(ii_output_dir, f"{left}.txt"), 'a', encoding='utf-8'))

        return [gr.update(), None]


def interrogate(image):
    prompt = shared.interrogator.interrogate(image.convert("RGB"))
    return gr.update() if prompt is None else prompt


def interrogate_deepbooru(image):
    prompt = deepbooru.model.tag(image)
    return gr.update() if prompt is None else prompt


def connect_clear_prompt(button):
    """Given clear button, prompt, and token_counter objects, setup clear prompt button click event"""
    button.click(
        _js="clear_prompt",
        fn=None,
        inputs=[],
        outputs=[],
    )


def update_token_counter(text, steps, styles, *, is_positive=True):
    params = script_callbacks.BeforeTokenCounterParams(text, steps, styles, is_positive=is_positive)
    script_callbacks.before_token_counter_callback(params)
    text = params.prompt
    steps = params.steps
    styles = params.styles
    is_positive = params.is_positive

    if shared.opts.include_styles_into_token_counters:
        apply_styles = shared.prompt_styles.apply_styles_to_prompt if is_positive else shared.prompt_styles.apply_negative_styles_to_prompt
        text = apply_styles(text, styles)

    try:
        text, _ = extra_networks.parse_prompt(text)

        if is_positive:
            _, prompt_flat_list, _ = prompt_parser.get_multicond_prompt_list([text])
        else:
            prompt_flat_list = [text]

        prompt_schedules = prompt_parser.get_learned_conditioning_prompt_schedules(prompt_flat_list, steps)

    except Exception:
        # a parsing error can happen here during typing, and we don't want to bother the user with
        # messages related to it in console
        prompt_schedules = [[[steps, text]]]

    flat_prompts = reduce(lambda list1, list2: list1+list2, prompt_schedules)
    prompts = [prompt_text for step, prompt_text in flat_prompts]
    token_count, max_length = max([model_hijack.get_prompt_lengths(prompt) for prompt in prompts], key=lambda args: args[0])
    return f"<span class='gr-box gr-text-input'>{token_count}/{max_length}</span>"


def update_negative_prompt_token_counter(*args):
    return update_token_counter(*args, is_positive=False)


def setup_progressbar(*args, **kwargs):
    pass


def apply_setting(key, value):
    if value is None:
        return gr.update()

    if shared.cmd_opts.freeze_settings:
        return gr.update()

    # dont allow model to be swapped when model hash exists in prompt
    if key == "sd_model_checkpoint" and opts.disable_weights_auto_swap:
        return gr.update()

    if key == "sd_model_checkpoint":
        ckpt_info = sd_models.get_closet_checkpoint_match(value)

        if ckpt_info is not None:
            value = ckpt_info.title
        else:
            return gr.update()

    comp_args = opts.data_labels[key].component_args
    if comp_args and isinstance(comp_args, dict) and comp_args.get('visible') is False:
        return

    valtype = type(opts.data_labels[key].default)
    oldval = opts.data.get(key, None)
    opts.data[key] = valtype(value) if valtype != type(None) else value
    if oldval != value and opts.data_labels[key].onchange is not None:
        opts.data_labels[key].onchange()

    opts.save(shared.config_filename)
    return getattr(opts, key)


def create_output_panel(tabname, outdir, toprow=None):
    return ui_common.create_output_panel(tabname, outdir, toprow)


def ordered_ui_categories():
    user_order = {x.strip(): i * 2 + 1 for i, x in enumerate(shared.opts.ui_reorder_list)}

    for _, category in sorted(enumerate(shared_items.ui_reorder_categories()), key=lambda x: user_order.get(x[1], x[0] * 2 + 0)):
        yield category


def create_override_settings_dropdown(tabname, row):
    dropdown = gr.Dropdown([], label="Override settings", visible=False, elem_id=f"{tabname}_override_settings", multiselect=True)

    dropdown.change(
        fn=lambda x: gr.Dropdown.update(visible=bool(x)),
        inputs=[dropdown],
        outputs=[dropdown],
    )

    return dropdown

def create_ui():
    import modules.img2img
    import modules.txt2img
    if shared.opts is None:
        import modules.initialize as initialize
        initialize.initialize() 

    reload_javascript()

    parameters_copypaste.reset()

    settings = ui_settings.UiSettings()
    settings.register_settings()

    scripts.scripts_current = scripts.scripts_txt2img
    scripts.scripts_txt2img.initialize_scripts(is_img2img=False)

    with gr.Blocks(analytics_enabled=False) as txt2img_interface:
        toprow = ui_toprow.Toprow(is_img2img=False, is_compact=shared.opts.compact_prompt_box)

        dummy_component = gr.Label(visible=False)

        extra_tabs = gr.Tabs(elem_id="txt2img_extra_tabs", elem_classes=["extra-networks"])
        extra_tabs.__enter__()

        with gr.Tab("Generation", id="txt2img_generation") as txt2img_generation_tab, ResizeHandleRow(equal_height=False):
            with ExitStack() as stack:
                if shared.opts.txt2img_settings_accordion:
                    stack.enter_context(gr.Accordion("Open for Settings", open=False))
                stack.enter_context(gr.Column(variant='compact', elem_id="txt2img_settings"))

                scripts.scripts_txt2img.prepare_ui()

                for category in ordered_ui_categories():
                    if category == "prompt":
                        toprow.create_inline_toprow_prompts()

                    elif category == "dimensions":
                        with FormRow():
                            with gr.Column(elem_id="txt2img_column_size", scale=4):
                                width = gr.Slider(minimum=64, maximum=2048, step=8, label="Width", value=1024, elem_id="txt2img_width")
                                height = gr.Slider(minimum=64, maximum=2048, step=8, label="Height", value=1024, elem_id="txt2img_height")

                            with gr.Column(elem_id="txt2img_dimensions_row", scale=1, elem_classes="dimensions-tools"):
                                res_switch_btn = ToolButton(value=switch_values_symbol, elem_id="txt2img_res_switch_btn", tooltip="Switch width/height")

                            if opts.dimensions_and_batch_together:
                                with gr.Column(elem_id="txt2img_column_batch"):
                                    batch_count = gr.Slider(minimum=1, step=1, label='Batch count', value=1, elem_id="txt2img_batch_count")
                                    batch_size = gr.Slider(minimum=1, maximum=8, step=1, label='Batch size', value=1, elem_id="txt2img_batch_size")

                    elif category == "cfg":
                        with gr.Row():
                            cfg_scale = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label='CFG Scale', value=7.0, elem_id="txt2img_cfg_scale")

                    elif category == "checkboxes":
                        with FormRow(elem_classes="checkboxes-row", variant="compact"):
                            pass

                    elif category == "accordions":
                        with gr.Row(elem_id="txt2img_accordions", elem_classes="accordions"):
                            with InputAccordion(False, label="Hires. fix", elem_id="txt2img_hr") as enable_hr:
                                with enable_hr.extra():
                                    hr_final_resolution = FormHTML(value="", elem_id="txtimg_hr_finalres", label="Upscaled resolution", interactive=False, min_width=0)

                                with FormRow(elem_id="txt2img_hires_fix_row1", variant="compact"):
                                    hr_upscaler = gr.Dropdown(label="Upscaler", elem_id="txt2img_hr_upscaler", choices=[*shared.latent_upscale_modes, *[x.name for x in shared.sd_upscalers]], value=shared.latent_upscale_default_mode)
                                    hr_second_pass_steps = gr.Slider(minimum=0, maximum=150, step=1, label='Hires steps', value=0, elem_id="txt2img_hires_steps")
                                    denoising_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Denoising strength', value=0.7, elem_id="txt2img_denoising_strength")

                                with FormRow(elem_id="txt2img_hires_fix_row2", variant="compact"):
                                    hr_scale = gr.Slider(minimum=1.0, maximum=4.0, step=0.05, label="Upscale by", value=2.0, elem_id="txt2img_hr_scale")
                                    hr_resize_x = gr.Slider(minimum=0, maximum=2048, step=8, label="Resize width to", value=0, elem_id="txt2img_hr_resize_x")
                                    hr_resize_y = gr.Slider(minimum=0, maximum=2048, step=8, label="Resize height to", value=0, elem_id="txt2img_hr_resize_y")

                                with FormRow(elem_id="txt2img_hires_fix_row3", variant="compact", visible=opts.hires_fix_show_sampler) as hr_sampler_container:

                                    hr_checkpoint_name = gr.Dropdown(label='Checkpoint', elem_id="hr_checkpoint", choices=["Use same checkpoint"] + modules.sd_models.checkpoint_tiles(use_short=True), value="Use same checkpoint")
                                    create_refresh_button(hr_checkpoint_name, modules.sd_models.list_models, lambda: {"choices": ["Use same checkpoint"] + modules.sd_models.checkpoint_tiles(use_short=True)}, "hr_checkpoint_refresh")

                                    hr_sampler_name = gr.Dropdown(label='Hires sampling method', elem_id="hr_sampler", choices=["Use same sampler"] + sd_samplers.visible_sampler_names(), value="Use same sampler")
                                    hr_scheduler = gr.Dropdown(label='Hires schedule type', elem_id="hr_scheduler", choices=["Use same scheduler"] + [x.label for x in sd_schedulers.schedulers], value="Use same scheduler")

                                with FormRow(elem_id="txt2img_hires_fix_row4", variant="compact", visible=opts.hires_fix_show_prompts) as hr_prompts_container:
                                    with gr.Column(scale=80):
                                        with gr.Row():
                                            hr_prompt = gr.Textbox(label="Hires prompt", elem_id="hires_prompt", show_label=False, lines=3, placeholder="Prompt for hires fix pass.\nLeave empty to use the same prompt as in first pass.", elem_classes=["prompt"])
                                    with gr.Column(scale=80):
                                        with gr.Row():
                                            hr_negative_prompt = gr.Textbox(label="Hires negative prompt", elem_id="hires_neg_prompt", show_label=False, lines=3, placeholder="Negative prompt for hires fix pass.\nLeave empty to use the same negative prompt as in first pass.", elem_classes=["prompt"])

                            scripts.scripts_txt2img.setup_ui_for_section(category)

                    elif category == "batch":
                        if not opts.dimensions_and_batch_together:
                            with FormRow(elem_id="txt2img_column_batch"):
                                batch_count = gr.Slider(minimum=1, step=1, label='Batch count', value=1, elem_id="txt2img_batch_count")
                                batch_size = gr.Slider(minimum=1, maximum=8, step=1, label='Batch size', value=1, elem_id="txt2img_batch_size")

                    elif category == "override_settings":
                        with FormRow(elem_id="txt2img_override_settings_row") as row:
                            override_settings = create_override_settings_dropdown('txt2img', row)

                    elif category == "scripts":
                        with FormGroup(elem_id="txt2img_script_container"):
                            custom_inputs = scripts.scripts_txt2img.setup_ui()

                    if category not in {"accordions"}:
                        scripts.scripts_txt2img.setup_ui_for_section(category)

            hr_resolution_preview_inputs = [enable_hr, width, height, hr_scale, hr_resize_x, hr_resize_y]

            for component in hr_resolution_preview_inputs:
                event = component.release if isinstance(component, gr.Slider) else component.change

                event(
                    fn=calc_resolution_hires,
                    inputs=hr_resolution_preview_inputs,
                    outputs=[hr_final_resolution],
                    show_progress=False,
                )
                event(
                    None,
                    _js="onCalcResolutionHires",
                    inputs=hr_resolution_preview_inputs,
                    outputs=[],
                    show_progress=False,
                )

            output_panel = create_output_panel("txt2img", opts.outdir_txt2img_samples, toprow)            

            txt2img_inputs = [
                dummy_component,
                toprow.prompt,
                toprow.negative_prompt,
                toprow.ui_styles.dropdown,
                batch_count,
                batch_size,
                cfg_scale,
                height,
                width,
                enable_hr,
                denoising_strength,
                hr_scale,
                hr_upscaler,
                hr_second_pass_steps,
                hr_resize_x,
                hr_resize_y,
                hr_checkpoint_name,
                hr_sampler_name,
                hr_scheduler,
                hr_prompt,
                hr_negative_prompt,
                override_settings,
            ] + custom_inputs


            txt2img_outputs = [
                output_panel.gallery,
                output_panel.generation_info,
                output_panel.infotext,
                output_panel.html_log,
            ]

            txt2img_args = dict(
                fn=wrap_gradio_gpu_call(modules.txt2img.txt2img, extra_outputs=[None, '', '']),
                _js="submit",
                inputs=txt2img_inputs,
                outputs=txt2img_outputs,
                show_progress=False,
            )

            toprow.prompt.submit(**txt2img_args)
            toprow.submit.click(**txt2img_args)


            output_panel.button_upscale.click(
                fn=wrap_gradio_gpu_call(modules.txt2img.txt2img_upscale, extra_outputs=[None, '', '']),
                _js="submit_txt2img_upscale",
                inputs=txt2img_inputs[0:1] + [output_panel.gallery, dummy_component, output_panel.generation_info] + txt2img_inputs[1:],
                outputs=txt2img_outputs,
                show_progress=False,
            )

            res_switch_btn.click(fn=None, _js="function(){switchWidthHeight('txt2img')}", inputs=None, outputs=None, show_progress=False)

            toprow.restore_progress_button.click(
                fn=progress.restore_progress,
                _js="restoreProgressTxt2img",
                inputs=[dummy_component],
                outputs=[
                    output_panel.gallery,
                    output_panel.generation_info,
                    output_panel.infotext,
                    output_panel.html_log,
                ],
                show_progress=False,
            )

            txt2img_paste_fields = [
                PasteField(toprow.prompt, "Prompt", api="prompt"),
                PasteField(toprow.negative_prompt, "Negative prompt", api="negative_prompt"),
                PasteField(cfg_scale, "CFG scale", api="cfg_scale"),
                PasteField(width, "Size-1", api="width"),
                PasteField(height, "Size-2", api="height"),
                PasteField(batch_size, "Batch size", api="batch_size"),
                PasteField(toprow.ui_styles.dropdown, lambda d: d["Styles array"] if isinstance(d.get("Styles array"), list) else gr.update(), api="styles"),
                PasteField(denoising_strength, "Denoising strength", api="denoising_strength"),
                PasteField(enable_hr, lambda d: "Denoising strength" in d and ("Hires upscale" in d or "Hires upscaler" in d or "Hires resize-1" in d), api="enable_hr"),
                PasteField(hr_scale, "Hires upscale", api="hr_scale"),
                PasteField(hr_upscaler, "Hires upscaler", api="hr_upscaler"),
                PasteField(hr_second_pass_steps, "Hires steps", api="hr_second_pass_steps"),
                PasteField(hr_resize_x, "Hires resize-1", api="hr_resize_x"),
                PasteField(hr_resize_y, "Hires resize-2", api="hr_resize_y"),
                PasteField(hr_checkpoint_name, "Hires checkpoint", api="hr_checkpoint_name"),
                PasteField(hr_sampler_name, sd_samplers.get_hr_sampler_from_infotext, api="hr_sampler_name"),
                PasteField(hr_scheduler, sd_samplers.get_hr_scheduler_from_infotext, api="hr_scheduler"),
                PasteField(hr_sampler_container, lambda d: gr.update(visible=True) if d.get("Hires sampler", "Use same sampler") != "Use same sampler" or d.get("Hires checkpoint", "Use same checkpoint") != "Use same checkpoint" or d.get("Hires schedule type", "Use same scheduler") != "Use same scheduler" else gr.update()),
                PasteField(hr_prompt, "Hires prompt", api="hr_prompt"),
                PasteField(hr_negative_prompt, "Hires negative prompt", api="hr_negative_prompt"),
                PasteField(hr_prompts_container, lambda d: gr.update(visible=True) if d.get("Hires prompt", "") != "" or d.get("Hires negative prompt", "") != "" else gr.update()),
                *scripts.scripts_txt2img.infotext_fields
            ]
            parameters_copypaste.add_paste_fields("txt2img", None, txt2img_paste_fields, override_settings)
            parameters_copypaste.register_paste_params_button(parameters_copypaste.ParamBinding(
                paste_button=toprow.paste, tabname="txt2img", source_text_component=toprow.prompt, source_image_component=None,
            ))

            steps = scripts.scripts_txt2img.script('Sampler').steps

            txt2img_preview_params = [
                toprow.prompt,
                toprow.negative_prompt,
                steps,
                scripts.scripts_txt2img.script('Sampler').sampler_name,
                cfg_scale,
                scripts.scripts_txt2img.script('Seed').seed,
                width,
                height,
            ]

            toprow.ui_styles.dropdown.change(fn=wrap_queued_call(update_token_counter), inputs=[toprow.prompt, steps, toprow.ui_styles.dropdown], outputs=[toprow.token_counter])
            toprow.ui_styles.dropdown.change(fn=wrap_queued_call(update_negative_prompt_token_counter), inputs=[toprow.negative_prompt, steps, toprow.ui_styles.dropdown], outputs=[toprow.negative_token_counter])
            toprow.token_button.click(fn=wrap_queued_call(update_token_counter), inputs=[toprow.prompt, steps, toprow.ui_styles.dropdown], outputs=[toprow.token_counter])
            toprow.negative_token_button.click(fn=wrap_queued_call(update_negative_prompt_token_counter), inputs=[toprow.negative_prompt, steps, toprow.ui_styles.dropdown], outputs=[toprow.negative_token_counter])

        extra_networks_ui = ui_extra_networks.create_ui(txt2img_interface, [txt2img_generation_tab], 'txt2img')
        ui_extra_networks.setup_ui(extra_networks_ui, output_panel.gallery)
        extra_tabs.__exit__()


    loadsave = ui_loadsave.UiLoadsave(cmd_opts.ui_config_file)
    ui_settings_from_file = loadsave.ui_settings.copy()

    settings.create_ui(loadsave, dummy_component)


    interfaces = [
        (txt2img_interface, "txt2img", "txt2img"),
        (settings.interface, "Settings", "settings"),
    ]

    interfaces += script_callbacks.ui_tabs_callback()
    extensions_interface = ui_extensions.create_ui()
    interfaces += [(extensions_interface, "Extensions", "extensions")]

    
    shared.tab_names = []
    for _interface, label, _ifid in interfaces:
        shared.tab_names.append(label)

    with gr.Blocks(theme=shared.gradio_theme, analytics_enabled=False, title="Stable Diffusion") as demo:
        settings.add_quicksettings()

        parameters_copypaste.connect_paste_params_buttons()

        with gr.Tabs(elem_id="tabs") as tabs:
            tab_order = {k: i for i, k in enumerate(opts.ui_tab_order)}
            sorted_interfaces = sorted(interfaces, key=lambda x: tab_order.get(x[1], 9999))

            for interface, label, ifid in sorted_interfaces:
                if label in shared.opts.hidden_tabs:
                    continue
                with gr.TabItem(label, id=ifid, elem_id=f"tab_{ifid}"):
                    interface.render()

                if ifid not in ["extensions", "settings"]:
                    loadsave.add_block(interface, ifid)

            loadsave.add_component(f"webui/Tabs@{tabs.elem_id}", tabs)

            loadsave.setup_ui()

        if os.path.exists(os.path.join(script_path, "notification.mp3")) and shared.opts.notification_audio:
            gr.Audio(interactive=False, value=os.path.join(script_path, "notification.mp3"), elem_id="audio_notification", visible=False)

        footer = shared.html("footer.html")
        footer = footer.format(versions=versions_html(), api_docs="/docs" if shared.cmd_opts.api else "https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/API")
        gr.HTML(footer, elem_id="footer")

        settings.add_functionality(demo)

    if ui_settings_from_file != loadsave.ui_settings:
        loadsave.dump_defaults()
    demo.ui_loadsave = loadsave

    return demo

def versions_html():
    import torch
    import launch

    python_version = ".".join([str(x) for x in sys.version_info[0:3]])
    commit = launch.commit_hash()
    tag = launch.git_tag()

    if shared.xformers_available:
        import xformers
        xformers_version = xformers.__version__
    else:
        xformers_version = "N/A"

    return f"""
version: <a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui/commit/{commit}">{tag}</a>
&#x2000;•&#x2000;
python: <span title="{sys.version}">{python_version}</span>
&#x2000;•&#x2000;
torch: {getattr(torch, '__long_version__',torch.__version__)}
&#x2000;•&#x2000;
xformers: {xformers_version}
&#x2000;•&#x2000;
gradio: {gr.__version__}
&#x2000;•&#x2000;
checkpoint: <a id="sd_checkpoint_hash">N/A</a>
"""


def setup_ui_api(app):
    from pydantic import BaseModel, Field

    class QuicksettingsHint(BaseModel):
        name: str = Field(title="Name of the quicksettings field")
        label: str = Field(title="Label of the quicksettings field")

    def quicksettings_hint():
        return [QuicksettingsHint(name=k, label=v.label) for k, v in opts.data_labels.items()]

    app.add_api_route("/internal/quicksettings-hint", quicksettings_hint, methods=["GET"], response_model=list[QuicksettingsHint])

    app.add_api_route("/internal/ping", lambda: {}, methods=["GET"])

    app.add_api_route("/internal/profile-startup", lambda: timer.startup_record, methods=["GET"])

    def download_sysinfo(attachment=False):
        from fastapi.responses import PlainTextResponse

        text = sysinfo.get()
        filename = f"sysinfo-{datetime.datetime.utcnow().strftime('%Y-%m-%d-%H-%M')}.json"

        return PlainTextResponse(text, headers={'Content-Disposition': f'{"attachment" if attachment else "inline"}; filename="{filename}"'})

    app.add_api_route("/internal/sysinfo", download_sysinfo, methods=["GET"])
    app.add_api_route("/internal/sysinfo-download", lambda: download_sysinfo(attachment=True), methods=["GET"])

    import fastapi.staticfiles
    app.mount("/webui-assets", fastapi.staticfiles.StaticFiles(directory=launch_utils.repo_dir('stable-diffusion-webui-assets')), name="webui-assets")