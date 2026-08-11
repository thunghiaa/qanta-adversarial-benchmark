import re
import threading

import gradio as gr
import pandas as pd
from loguru import logger

from backend import eval_suite
from backend.jobs import JobStage, JobStatus, JobTask
from display.formatting import styled_error, styled_message
from envs import EVAL_SPLITS
from main_backend import download_submissions_and_create_jobs, run_next_job
from utils import (
    JOB_STAGES,
    JOB_STATUSES,
    STATUS_LABELS,
    generate_header_dashboard,
    generate_job_info_card,
    get_all_jobs_df,
    job_manager,
    stage_labels,
    stage_mini_labels,
    status_colors,
    status_emojis,
    task_labels,
    tasks,
)

STATUS_CHOICES = [f"{status_emojis[js]}" for js in JobStatus]
STATUS_DEFAULT_SELECTIONS = [f"{status_emojis[js]}" for js in [JobStatus.SUBMITTED, JobStatus.FAILED]]


def view_logs(request_id):
    # TODO: implement this
    # Replace with actual function to retrieve logs for a specific job
    return f"Displaying logs for job {request_id}...\n\n CURRENTLY NOT IMPLEMENTED."


def extract_job_id_from_html(html_str: str) -> str:
    return re.sub(r"<.*?>", "", html_str)


def styled_job_status(status: JobStatus) -> str:
    return f"<p style='color: {status_colors[status]}; font-weight: bold;'>{status_emojis[status]} {STATUS_LABELS[status]}</p>"


def re_evaluate_elo(eval_split: str):
    logger.info(f"Re-evaluating ELO for split {eval_split}")
    eval_suite.request_elo_computation(eval_split)
    threading.Thread(target=eval_suite.check_and_run_elo_computation, args=(eval_split,), daemon=True).start()
    return styled_message(f"ELO computation requested for split {eval_split}")


def resubmit_job(job_id, job_stage_index: int | None):
    # Replace with actual resubmission logic
    logger.info(f"Resubmitting job {job_id}")
    if not job_id:
        logger.error("No job ID provided")
        return styled_error("No job ID provided")

    if job_stage_index is None or job_stage_index == 0:
        job = job_manager.get_job(job_id)
        if job is not None:
            try:
                job_stage_index = JOB_STAGES.index(job.stage) + 1
            except ValueError:
                job_stage_index = 0
    if job_stage_index is None or job_stage_index == 0:
        logger.error("No job stage provided")
        return styled_error(
            "No job stage provided — pick a job from the table (or set Job ID) so the stage is known."
        )
    try:
        stage = JOB_STAGES[job_stage_index - 1]
        job_manager.update_job_stage(job_id, stage, JobStatus.SUBMITTED, message=f"Resubmitted from {stage}")
        return styled_message(f"Job {job_id} resubmitted successfully")
    except FileNotFoundError as e:
        logger.error(f"Error resubmitting job {job_id}: {e}")
        return styled_error(f"Job not found: '{job_id}'")


def delete_job(job_id):
    if not job_id:
        logger.error("No job ID provided")
        return styled_error("No job ID provided")
    try:
        job_manager.remove_job(job_id)
        return styled_message(f"Job {job_id} deleted successfully")
    except FileNotFoundError:
        return styled_error(f"Job not found: '{job_id}'")


def batch_resubmit_jobs(jobs_df: pd.DataFrame):
    job_ids = jobs_df["JobID"].apply(extract_job_id_from_html).tolist()
    for job_id in job_ids:
        # remomve html tags from the job id
        job_id = extract_job_id_from_html(job_id)
        job = job_manager.get_job(job_id)
        logger.info(f"Resubmitting job {job_id} from {job.stage}")
        job_manager.update_job_stage(job, job.stage, JobStatus.SUBMITTED, message=f"Resubmitted for {job.stage}")


def filter_jobs_df(df: pd.DataFrame, stage_filter: int, status_filter: list[int], task_filter: int):
    if stage_filter > 0:
        stage = JOB_STAGES[stage_filter - 1]
        df = df[df["Stage"] == stage.lower()]
    if status_filter:
        statuses = [JOB_STATUSES[i] for i in status_filter]
        df = df[df["Status"].isin(statuses)]
    if task_filter > 0:
        task = tasks[task_filter - 1]
        df = df[df["Task"] == task.lower()]
    df.loc[:, "Task"] = df["Task"].apply(lambda x: task_labels[x])
    df.loc[:, "Status"] = df["Status"].apply(styled_job_status)
    df.loc[:, "Stage"] = df["Stage"].apply(lambda x: f"{stage_mini_labels[x]}")
    return df


def filter_jobs_dataframe(stage_filter: int, status_filter: list[int], task_filter: int):
    df = get_all_jobs_df()
    df = filter_jobs_df(df, stage_filter, status_filter, task_filter)
    return df, gr.update(choices=df["JobID"].apply(extract_job_id_from_html).tolist())


def refresh_interface_data(stage_filter: int, status_filter: list[int], task_filter: int):
    jobs_df = get_all_jobs_df()
    header_dash = generate_header_dashboard(jobs_df)
    jobs_df = filter_jobs_df(jobs_df, stage_filter, status_filter, task_filter)
    return header_dash, jobs_df, gr.update(choices=jobs_df["JobID"].apply(extract_job_id_from_html).tolist())


def render_job_card(job_id):
    # job_stage_selector uses type="index" with a leading "" choice; value must be int index.

    if not job_id or job_id.strip() == "":
        return (
            gr.HTML(""),
            gr.update(interactive=False, value=""),
            gr.update(interactive=False),
        )
    try:
        job = job_manager.get_job(job_id)
        if not job:
            raise FileNotFoundError(f"Job not found: '{job_id}'")
    except FileNotFoundError:
        return (
            gr.HTML(f"Job not found: '{job_id}'"),
            gr.update(interactive=False, value=""),
            gr.update(interactive=False),
        )
    job_info_html = generate_job_info_card(job)
    # choices are ["", outputs, metrics, evaluation] -> current stage is at JOB_STAGES.index + 1
    return (
        gr.update(value=job_info_html),
        gr.update(value=stage_mini_labels[job.stage], interactive=True),
        gr.update(interactive=True),
    )


def create_requests_dashboard(app, refresh_btn):
    jobs_table = get_all_jobs_df()

    # ---------------------------------------------* UI COMPONENTS *---------------------------------------------------
    header_html = gr.HTML(generate_header_dashboard(jobs_table))

    with gr.Row():
        stage_choices = ["ALL"] + [f"{stage_labels[stage]}" for stage in JobStage]
        task_choices = ["ALL"] + [f"{task_labels[task]}" for task in JobTask]
        job_status_filter = gr.Dropdown(
            choices=STATUS_CHOICES,
            label="Job Status",
            info="Filter by status of the job",
            type="index",
            multiselect=True,
            value=STATUS_DEFAULT_SELECTIONS,
            interactive=True,
            scale=2,
        )
        job_stage_filter = gr.Dropdown(
            choices=stage_choices,
            label="Job Stage",
            info="Filter by stage of the job",
            type="index",
            value=stage_choices[0],
            interactive=True,
            scale=1,
        )
        job_task_filter = gr.Dropdown(
            choices=task_choices,
            label="Job Task",
            info="Filter by competition type (Task)",
            type="index",
            value=task_choices[0],
            interactive=True,
            scale=1,
        )
        with gr.Column(scale=1):
            batch_resubmit_btn = gr.Button(
                "📤 Resubmit Jobs", size="lg", variant="primary", scale=1, elem_classes="my-btn"
            )
            with gr.Column(visible=False) as confirm_dialog:
                confirm_message = gr.Markdown("#### **Are you sure you want to resubmit all jobs?**")
                with gr.Row():
                    confirm_btn = gr.Button("👍🏾 Confirm", variant="secondary", size="md", scale=0, min_width=100)
                    cancel_btn = gr.Button("❌ Cancel", variant="secondary", size="md", scale=0, min_width=100)

    job_id_choices = [""] + jobs_table["JobID"].apply(extract_job_id_from_html).tolist()
    jobs_table = gr.DataFrame(
        jobs_table,
        label="QANTA 2025 Evaluation Queue",
        headers=["JobID", "Status", "Stage", "Task", "User", "Model", "Timestamp (UTC)", "Split", "Message"],
        datatype=["markdown", "markdown", "str", "str", "markdown", "markdown", "str", "str", "str"],
        interactive=False,
        max_height=300,
        # show_row_numbers=True,
        column_widths=[4, 6, 8, 6, 8, 10, 8, 5, 14],
        show_search="filter",
    )

    with gr.Row():
        with gr.Column():
            job_id_selector = gr.Dropdown(
                choices=job_id_choices,
                label="Job ID",
                info="Select the job you want to interact with",
                allow_custom_value=True,
                interactive=True,
            )
            job_stage_selector = gr.Dropdown(
                choices=[""] + [f"{stage_mini_labels[stage]}" for stage in JobStage],
                label="Job Stage",
                info="Which stage you want to resubmit",
                type="index",
                allow_custom_value=True,
            )
            with gr.Row():
                trigger_cb = gr.Checkbox(label="Trigger Now", min_width=80)
                resubmit_btn = gr.Button("📤 Resubmit Job")
            with gr.Row():
                view_logs_btn = gr.Button("🔍 View Logs")
                delete_btn = gr.Button("🗑️ Delete Job")
            with gr.Row():
                split_selector = gr.Dropdown(
                    choices=EVAL_SPLITS,
                    label="Eval Split",
                    info="Which split you want to re-evaluate ELO for?",
                    type="value",
                    value=EVAL_SPLITS[0],
                    scale=1,
                )
                re_eval_elo_btn = gr.Button("🔄 Re-evaluate ELO", scale=1, min_width=100)

        with gr.Column():
            job_info = gr.HTML()
            resubmit_output = gr.HTML()

    gr.Markdown("## Manual Triggers")
    with gr.Row():
        jobs_button = gr.Button("📤 Create Jobs")
        out_button = gr.Button("🛠️ Trigger Outputs Gen")
        metrics_button = gr.Button("📊 Trigger Metrics Gen")
        eval_button = gr.Button("🏆 Trigger Evaluation")

    log_output = gr.Textbox(label="Job Logs", lines=10)

    # --------------------------------------------* EVENT LISTENERS *--------------------------------------------------
    gr.on(
        triggers=[job_status_filter.change, job_task_filter.change, job_stage_filter.change],
        fn=filter_jobs_dataframe,
        inputs=[job_stage_filter, job_status_filter, job_task_filter],
        outputs=[jobs_table, job_id_selector],
    )

    def ask_batch_resubmit_confirmation(jobs_df: pd.DataFrame):
        msg = f"Are you sure you want to resubmit all {len(jobs_df)} jobs?"
        return gr.update(visible=True), msg

    interface_refresh_trigger_dict = {
        "fn": refresh_interface_data,
        "inputs": [job_stage_filter, job_status_filter, job_task_filter],
        "outputs": [header_html, jobs_table, job_id_selector],
    }

    batch_resubmit_btn.click(
        fn=ask_batch_resubmit_confirmation, inputs=[jobs_table], outputs=[confirm_dialog, confirm_message]
    )
    cancel_btn.click(fn=lambda: (gr.update(visible=False), ""), outputs=[confirm_dialog, confirm_message])
    confirm_btn.click(fn=batch_resubmit_jobs, inputs=[jobs_table]).then(
        fn=lambda: (gr.update(visible=False), ""), outputs=[confirm_dialog, confirm_message]
    ).then(**interface_refresh_trigger_dict)

    def df_select_callback(df: pd.DataFrame, evt: gr.SelectData):
        job_id = extract_job_id_from_html(evt.row_value[0])
        return gr.update(value=job_id)

    job_card_render_dict = {
        "fn": render_job_card,
        "inputs": [job_id_selector],
        "outputs": [job_info, job_stage_selector, resubmit_btn],
    }
    jobs_table.select(fn=df_select_callback, inputs=[jobs_table], outputs=[job_id_selector]).success(
        **job_card_render_dict
    )
    job_id_selector.select(**job_card_render_dict)
    view_logs_btn.click(fn=view_logs, inputs=[job_id_selector], outputs=[log_output])
    resubmit_btn.click(fn=resubmit_job, inputs=[job_id_selector, job_stage_selector], outputs=[resubmit_output]).then(
        **interface_refresh_trigger_dict
    ).success(
        fn=lambda idx, trigger: run_next_job(JOB_STAGES[idx - 1]) if trigger else None,
        inputs=[job_stage_selector, trigger_cb],
    )
    delete_btn.click(fn=delete_job, inputs=[job_id_selector], outputs=[resubmit_output]).then(
        **interface_refresh_trigger_dict
    )

    # ELO Re-evaluation
    re_eval_elo_btn.click(fn=re_evaluate_elo, inputs=[split_selector], outputs=[resubmit_output])

    gr.on(triggers=[app.load, refresh_btn.click], **interface_refresh_trigger_dict)
    out_button.click(fn=run_next_job, inputs=[gr.State(JobStage.OUTPUTS)], outputs=[])
    metrics_button.click(fn=run_next_job, inputs=[gr.State(JobStage.METRICS)], outputs=[])
    eval_button.click(fn=run_next_job, inputs=[gr.State(JobStage.EVALUATION)], outputs=[])
    jobs_button.click(fn=download_submissions_and_create_jobs, inputs=[], outputs=[], concurrency_limit=1)
