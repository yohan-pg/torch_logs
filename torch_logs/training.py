from .imports import *

import torch_logs.logger as logger

from .utils import capture_text_output, latest_numbered_directory


class LogEvent(enum.Enum):
    INIT = "INIT"
    RESUME = "RESUME"
    TICK = "TICK"
    PROGRESS = "PROGRESS"
    CHECKPOINT = "CHECKPOINT"
    EPOCH = "EPOCH"


@dataclass(frozen=True)
class Training(Generic[Preds]):
    path: str
    comment: Optional[str]
    max_iters: int

    training_loop: TrainingLoop
    validation_loop: Optional[EvaluationLoop]

    tick_frequency: int = 20
    progress_frequency: int = 200
    checkpoint_frequency: int = 1000

    save_prediction: Callable[[Any, str], None] = save_image

    def __iter__(
        self,
    ) -> Iterator[LogEvent]:
        with logger.directory(self.path) as exists_already:
            info("Starting training")

            if exists_already:
                self.rollback()
                yield LogEvent.RESUME
            else:
                self.init_logging()
                yield LogEvent.INIT

            i = 0

            with logger.progress(self.max_iters) as log_progress:
                while True:
                    for _ in self.training_loop:
                        if i >= self.max_iters:
                            break

                        if i % self.tick_frequency == 0:
                            self.tick_logging()
                            yield LogEvent.TICK

                        if i % self.progress_frequency == 0:
                            self.progress_logging()
                            yield LogEvent.PROGRESS

                        if i % self.checkpoint_frequency == 0:
                            self.checkpoint_logging()
                            yield LogEvent.CHECKPOINT

                        i += 1
                        log_progress(i)

                    self.epoch_logging()
                    yield LogEvent.EPOCH

    def latest_checkpoint_path(self):
        num = latest_numbered_directory(self.path + "/checkpoints")
        assert num != -1
        return f"checkpoints/{num:05d}"

    @torch.no_grad()
    def init_logging(self) -> None:
        info("Logging init")

        if "pytest" not in sys.argv[0]:  # todo figure out why this fails in tests
            capture_text_output()
        else:
            info("In pytest - log capture disabled")

        logger.log_pid()
        logger.log_config()
        logger.log_max_memory()

        logger.log_architecture(self.training_loop.model)

        if self.comment is not None:
            logger.log_comment(self.comment)

        logger.create_symbolic_link(self.path)

    @torch.no_grad()
    def tick_logging(self) -> None:
        info("Logging tick")

        train_batch = next(iter(self.training_loop.dataloader))
        logger.log_losses(self.training_loop.evaluation(train_batch))

        if self.validation_loop is not None:
            validation_batch = next(iter(self.validation_loop.dataloader))
            logger.log_losses(
                self.training_loop.evaluation(validation_batch), validation=True
            )

        logger.log_max_memory()

    @torch.no_grad()
    def progress_logging(self) -> None:
        info("Logging progress")

        logger.plot_losses(self.progress_frequency)

    @torch.no_grad()
    def checkpoint_logging(self) -> None:
        info("Logging checkpoint")

        scores = None

        with self._atomic_checkpoint_dir():
            logger.log_model(
                self.validation_loop.model
                if self.validation_loop is not None
                else self.training_loop.model
            )
            logger.log_training(self)

            if self.validation_loop is not None:
                scores = self._run_validation_loop()

        if scores is not None:
            logger.log_scores(scores)

        logger.plot_scores()


    def epoch_logging(self) -> None:
        assert self.training_loop.last_epoch_losses is not None
        logger.log_epoch_losses(self.training_loop.last_epoch_losses)
        logger.plot_epoch_losses()


    def _run_validation_loop(self) -> Scores:
        assert self.validation_loop is not None

        with logger.progress(len(self.validation_loop)) as log_progress:
            scores = None
            with logger.directory("predictions"):
                for j, (preds, scores) in enumerate(self.validation_loop):
                    logger.log_predictions(j, preds, self.save_prediction)
                    log_progress(j)

        assert scores is not None
        return scores

    def _backup_losses(self):
        """Backup losses into the checkpoint so we can restore them on resume"""
        shutil.copyfile("../../training_losses.csv", "training_losses.csv")
        shutil.copyfile("../../validation_losses.csv", "validation_losses.csv")

    @contextlib.contextmanager
    def _atomic_checkpoint_dir(self):
        """
        Create a numbered directory for the checkpoint,
        marking it as incomplete if the checkpoint fails partway.
        """
        try:
            # * The calls are split in two so that the empty parent directory exists the first time around
            with logger.directory("checkpoints"):
                num = latest_numbered_directory(".") + 1
                with logger.directory(f"{num:05d}"):
                    yield
        except Exception as e:
            try:
                path = self.latest_checkpoint_path()
                shutil.rmtree(f"{path}.incomplete")  # type: ignore
                shutil.move(f"{path}", f"{path}.incomplete")  # type: ignore
            except UnboundLocalError:
                pass
            except AssertionError:
                pass
            raise e

    def rollback(self):
        info("Performing rollback")
        try:
            rollback_path = self.latest_checkpoint_path()
        except AssertionError:
            info("No checkpoint found - training normally")
        else:
            warn(f"Resuming from {rollback_path}")
            # Discard the lost iterations from the losses
            shutil.copyfile(f"{path}/training_losses.csv", "training_losses.csv")
            shutil.copyfile(f"{path}/validation_losses.csv", "validation_losses.csv")
            logger.plot_losses(self.progress_frequency)

