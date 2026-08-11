from datetime import datetime

import pandas as pd

from display.formatting import model_hyperlink
from src.backend.jobs import Job, JobManager, JobStage, JobStatus, JobTask
from src.backend.submissions import SubmissionManager
from src.envs import API, JOBS_REPO, LOCAL_JOBS_PATH, LOCAL_REQUESTS_PATH, REQUESTS_REPO

submissions_manager = SubmissionManager(LOCAL_REQUESTS_PATH, REQUESTS_REPO, API)
job_manager = JobManager(LOCAL_JOBS_PATH, JOBS_REPO, submissions_manager, API)
JOB_STAGES = list(JobStage)
JOB_STATUSES = list(JobStatus)
tasks = list(JobTask)

task_labels = {
    "tossup": "🛎️ Tossup",
    "bonus": "🧐 Bonus",
}
STATUS_LABELS = {
    JobStatus.SUBMITTED: "Submitted",
    JobStatus.IN_PROGRESS: "In Progress",
    JobStatus.COMPLETED: "Completed",
    JobStatus.FAILED: "Failed",
}
stage_labels = {
    JobStage.OUTPUTS: "🛠️ Output Generation",
    JobStage.METRICS: "📊 Metrics Generation",
    JobStage.EVALUATION: "🏆 Evaluation (Leaderboard)",
}
stage_mini_labels = {
    JobStage.OUTPUTS: "🛠️ Outputs Gen",
    JobStage.METRICS: "📊 Metrics Gen",
    JobStage.EVALUATION: "🏆 Leaderboard",
}
status_colors = {
    JobStatus.SUBMITTED: "#3498db",
    JobStatus.IN_PROGRESS: "#f39c12",
    JobStatus.COMPLETED: "#2ecc71",
    JobStatus.FAILED: "#e74c3c",
}
status_emojis = {
    JobStatus.SUBMITTED: "📨",
    JobStatus.IN_PROGRESS: "🔄",
    JobStatus.COMPLETED: "✅",
    JobStatus.FAILED: "❌",
}


def get_all_jobs_df() -> pd.DataFrame:
    jobs = job_manager.get_jobs()

    def hyperlinked_model(job) -> str:
        return model_hyperlink(
            job_manager.get_submission_remote_url(job.id),
            job.model_name,
        )

    def hyperlinked_job_id(job) -> str:
        return model_hyperlink(
            job_manager.get_remote_url(job.id),
            job.id,
        )

    def hyperlinked_user(job) -> str:
        return model_hyperlink(
            job_manager.get_remote_user_url(job.username),
            job.username,
        )

    dt_format = "%B %d, %H:%M" if datetime.now().year == 2025 else "%Y-%m-%d %H:%M"

    df = pd.DataFrame(
        [
            {
                "JobID": hyperlinked_job_id(j),
                "Status": j.status,
                "Stage": j.stage,
                "Task": j.competition_type,
                "User": hyperlinked_user(j),
                "Model": hyperlinked_model(j),
                "Timestamp (UTC)": pd.to_datetime(j.created_at, utc=True).strftime(dt_format),
                "Split": j.eval_split,
                "Message": j.error,
            }
            for j in jobs
        ],
        columns=["JobID", "Status", "Stage", "Task", "User", "Model", "Timestamp (UTC)", "Split", "Message"],
    )
    return df


def generate_header_dashboard(jobs_df: pd.DataFrame):
    # Count jobs for each (stage, status)
    counts = {(stage, status): 0 for stage in JOB_STAGES for status in JOB_STATUSES}
    if len(jobs_df) > 0:
        for stage in JOB_STAGES:
            for status in JOB_STATUSES:
                counts[(stage, status)] = len(jobs_df[(jobs_df["Stage"] == stage) & (jobs_df["Status"] == status)])

    html = """
    <div class='pipeline-container'>
    """
    for stage in JOB_STAGES:
        html += "<div class='stage-card'>"
        html += f"<div class='stage-title'>{stage_labels[stage]}</div>"
        html += "<div class='status-row'>"
        for status in JOB_STATUSES:
            color = status_colors[status]
            emoji = status_emojis[status]
            count = counts[(stage, status)]
            html += (
                f"<div class='status-block' style='border-color: {color}; background: {color}10;'>"
                f"<span class='emoji'>{emoji}</span>"
                f"<span class='count' style='color: {color};'>{count}</span>"
                f"<span class='label'>{STATUS_LABELS[status]}</span>"
                f"</div>"
            )
        html += "</div>"  # status-row
        html += "</div>"  # stage-card
    html += "</div>"  # pipeline-container
    return html


def generate_job_info_card(job: Job) -> str:
    """
    Generate a beautiful HTML card for a job's info.
    Args:
        job (dict): Dictionary with job details.
    Returns:
        str: HTML string for the job info card.
    """
    # Extract and format fields
    job_id = job.id
    status = job.status
    stage = job.stage
    task = job.competition_type
    user = job.username
    model = job.model_name
    split = job.eval_split
    timestamp = pd.to_datetime(job.created_at).strftime("%Y-%m-%d %H:%M")
    error = job.error

    # Try to get pretty labels/emojis if possible
    try:
        status_label = STATUS_LABELS[status]
        status_emoji = status_emojis[status]
        status_color = status_colors[status]
    except Exception:
        status_label = status
        status_emoji = ""
        status_color = "#888"
    try:
        stage_label = stage_labels[stage]
        stage_emoji = stage_mini_labels[stage][0:2]
    except Exception:
        stage_label = stage
        stage_emoji = ""
    try:
        task_label = task_labels[task]
    except Exception:
        task_label = task

    card_html = f"""
    <div style="max-width: 420px; margin: 0 auto; background: #fff; border-radius: 18px; box-shadow: 0 4px 24px #0001; padding: 2rem 1.5rem 1.5rem 1.5rem; font-family: 'Inter', sans-serif;">
      <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;">
        <span style="font-size: 2rem;">{status_emoji}</span>
        <span style="font-size: 1.2rem; font-weight: 600; color: {status_color};">{status_label}</span>
        <span style="margin-left: auto; font-size: 0.95rem; color: #888;">{timestamp}</span>
      </div>
      <div style="margin-bottom: 0.7rem;">
        <span style="font-size: 1.1rem; font-weight: 500; color: #222;">Job ID:</span>
        <span style="font-family: monospace; color: #555;">{job_id}</span>
      </div>
      <div style="display: flex; gap: 1.2rem; margin-bottom: 0.7rem;">
        <div><span style="font-weight: 500;">Stage:</span> <span>{stage_emoji} {stage_label}</span></div>
        <div><span style="font-weight: 500;">Task:</span> <span>{task_label}</span></div>
      </div>
      <div style="display: flex; gap: 1.2rem; margin-bottom: 0.7rem;">
        <div><span style="font-weight: 500;">User:</span> <span>{user}</span></div>
        <div><span style="font-weight: 500;">Split:</span> <span>{split}</span></div>
      </div>
      <div style="margin-bottom: 0.7rem;"><span style="font-weight: 500;">Model:</span> <span>{model}</span></div>
      {f'<div style="margin-top: 1rem; color: #e74c3c; font-size: 0.98rem; background: #fbeaea; border-radius: 8px; padding: 0.5rem 0.8rem;"><b>Error:</b> {error}</div>' if error else ""}
    </div>
    """
    return card_html
