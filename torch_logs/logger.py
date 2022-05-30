from .imports import *


@contextlib.contextmanager
def directory(path: str) -> Iterable[bool]:
    relpath = os.path.relpath(os.getcwd() + "/" + path, os.path.dirname(main.__file__))
    info(f"- Opening directory '{relpath}'")

    exists_already = os.path.exists(path)
    if not exists_already:
        os.makedirs(path)

    cwd = os.getcwd()
    os.chdir(path)

    try:
        yield exists_already
    finally:
        os.chdir(cwd)


def log_comment(comment: str):
    info(f"- Logging comment")

    latest_comment_path = "../latest/comment.txt"
    if os.path.exists("../latest/comment.txt"):
        with open(latest_comment_path, "r") as f:
            assert (
                f.read() != comment
            ), "Comment did not change between runs! Write a new comment or delete the old one."

    with open("comment.txt", "w") as f:
        f.write(comment)


def log_architecture(model: Any):
    info(f"- Logging architecture")

    with open("architecture.txt", "w") as file:
        print(str(model), file=file)
        print("--------------------------------", file=file)
        temp = builtins.repr
        builtins.repr = nn.Module.__repr__
        print(repr(model), file=file)
        builtins.repr = temp


def log_pid() -> None:
    info(f"- Logging PID")

    with open("pid.txt", "w") as f:
        f.write(str(os.getpid()))


def log_max_memory() -> None:
    info(f"- Logging max memory usage")

    with open("max_memory.txt", "w") as f:
        f.write(
            str(torch.cuda.max_memory_allocated())
        )  # todo generalize to multiple devices


def log_config() -> None:
    info(f"- Logging launch configuration")

    shutil.copy(main.__file__, "config.py")


def log_losses(losses: Losses, validation: bool = False) -> None:
    info(f"- Logging {'validation' if validation else 'training'} losses")
    _write_losses_to_csv(losses, path = "training_losses.csv" if not validation else "validation_losses.csv")


def log_epoch_losses(losses: Losses, validation: bool = False) -> None:
    info(f"- Logging end of epoch losses")
    _write_losses_to_csv(losses, "epoch_losses.csv")


def _write_losses_to_csv(losses: Losses, path: str):
    if not os.path.exists(path):
        with open(path, "w") as file:
            print(",".join(list(losses.keys())), file=file)

    with open(path, "a") as file:
        print(",".join([str(float(value)) for value in losses.values()]), file=file)


def log_training(training: Any) -> None:
    info(f"- Logging training")

    torch.save(training, "training.pt")


def log_model(model: Model) -> None:
    info(f"- Logging model")
    torch.save(model, "model.pt")

    if isinstance(model, nn.Module):
        torch.save(model.state_dict(), "weights.pt")


def plot_losses(iters_per_tick: int) -> None:
    info(f"- Plotting losses")

    training_losses = pd.read_csv("training_losses.csv")
    validation_losses = (
        pd.read_csv("validation_losses.csv")
        if os.path.exists("validation_losses.csv")
        else None
    )
    assert (training_losses.columns == validation_losses.columns).all()

    os.makedirs("plots", exist_ok=True)

    images = []

    for col in training_losses.columns:
        if col == "iteration":
            continue
        lines = {
            "training": training_losses[col],
        }
        if validation_losses is not None:
            lines["validation"] = validation_losses[col]
        df = pd.DataFrame(lines)
        df.index = pd.RangeIndex(
            df.index.start, df.index.stop * iters_per_tick, step=iters_per_tick
        )
        fig = (
            px.line(
                df,
                title=f"Total weighted objective per iteration"
                if col == "objective"
                else f"Average '{col}' loss per iteration",
                y=list(lines.keys()),
                log_y=True,
                template="plotly_dark",
            )
            .update_layout(
                showlegend=validation_losses is not None, xaxis_title="iteration"
            )
            .update_traces(line=dict(width=4.0 if col == "objective" else 2.0))
        )
        fig.write_image(f"plots/losses_{col}.png")
        fig.write_html(f"plots/losses_{col}.html")
        images.append(read_image(f"plots/losses_{col}.png"))

    save_image(torch.stack(images) / 255, "losses_plot.png", padding=0, nrow=1)


def log_scores(scores: dict[str, float]) -> None:
    info(f"- Logging scores")

    path = "scores.csv"

    if not os.path.exists(path):
        with open(path, "w") as file:
            print(",".join(list(scores.keys())), file=file)

    with open(path, "a") as file:
        print(",".join([str(value) for value in scores.values()]), file=file)


def plot_epoch_losses() -> None:
    info(f"- Plotting end of epoch losses")

    scores = pd.read_csv("epoch_losses.csv")
    os.makedirs("plots", exist_ok=True)

    images = []

    for col in scores.columns:
        xs = list(range(1, len(scores[col]) + 1))
        fig = (
            px.line(
                scores[col],
                x=xs,
                y=col,
                title=f"'{col}' total training losses per epoch",
                log_y=True,
                template="plotly_dark",
            )
            .update_traces(mode="lines+markers")
            .update_layout(showlegend=False, xaxis_title="epoch")
            .update_xaxes(tickmode="array", tickvals=xs)
        )

        fig.write_image(f"plots/epoch_{col}.png")
        fig.write_html(f"plots/epoch_{col}.html")

        images.append(read_image(f"plots/epoch_{col}.png"))

    save_image(torch.stack(images) / 255, "epoch_plot.png", padding=0, nrow=1)


def plot_scores() -> None:
    info(f"- Plotting scores")

    scores = pd.read_csv("scores.csv")
    os.makedirs("plots", exist_ok=True)

    images = []

    for col in scores.columns:
        xs = list(range(1, len(scores[col]) + 1))
        fig = (
            px.line(
                scores[col],
                x=xs,
                y=col,
                title=f"'{col}' score per checkpoint",
                log_y=True,
                template="plotly_dark",
            )
            .update_traces(mode="lines+markers")
            .update_layout(showlegend=False, xaxis_title="checkpoint")
            .update_xaxes(tickmode="array", tickvals=xs)
        )

        fig.write_image(f"plots/scores_{col}.png")
        fig.write_html(f"plots/scores_{col}.html")

        images.append(read_image(f"plots/scores_{col}.png"))

    save_image(torch.stack(images) / 255, "scores_plot.png", padding=0, nrow=1)


def log_predictions(
    j: int,
    preds: Iterable,
    save_prediction: Callable[[Any, str], None],
    suffix: str = "",
):
    if j == 0:
        info(f"- Logging predictions")

    if isinstance(preds, dict):
        for key, value in enumerate(preds.items()):
            log_predictions(j, value, save_prediction, suffix=suffix + "_" + str(key))
    elif isinstance(preds, list) or isinstance(preds, tuple):
        for i, item in enumerate(preds):
            log_predictions(j, item, save_prediction, suffix=suffix + "_" + str(i))
    elif isinstance(preds, torch.Tensor):
        for k, item in enumerate(preds):
            num = f"{j*preds.shape[0]+k:06d}"
            directory = num[:4]
            os.makedirs(directory, exist_ok=True)
            save_prediction(
                item.unsqueeze(0),
                f"{directory}/{num}{'_' if suffix != '' else ''}{suffix}.png",
            )
    else:
        warn(f"Unable to log prediction: {preds}")


def create_symbolic_link(name: str):
    info(f"- Updating (or creating) symbolic link to the latest training")

    with directory(".."):
        if (
            "latest" in os.listdir()
        ):  # * os.path.latest does not work when the link is broken!
            os.remove("latest")
        os.symlink(name.split("/")[-1], "latest")


@contextlib.contextmanager
def progress(max_iters: int):
    path = os.path.abspath("progress.txt")

    def log_progress(i: int):
        with open(path, "w") as file:
            print(datetime.now().isoformat(timespec="seconds"), file=file)
            print(str(round((i / (max_iters - 1)) * 100, 4)) + "%", file=file)

    yield log_progress
