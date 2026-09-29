from fractaltokenizer_v2.imagenet_training import TrainingConfig, _alerts
from telegram_training_helper import TelegramBot


def test_diagnostic_alerts():
    config = TrainingConfig(
        collapse_epoch=5,
        collapse_train_top1=2.0,
        underfit_epoch=10,
        underfit_train_top1=30.0,
        overfit_gap=15.0,
        stagnation_patience=3,
    )
    history = [
        {"epoch": epoch, "train_top1": 1.0, "val_top1": 1.0, "val_loss": 4.6}
        for epoch in range(1, 11)
    ]
    alerts = _alerts(history, config)
    assert any("collapse" in alert for alert in alerts)
    assert any("underfitting" in alert for alert in alerts)
    assert any("stagnated" in alert for alert in alerts)


def test_telegram_terminal_commands_can_be_disabled():
    bot = TelegramBot(
        bot_token="dummy",
        chat_id="1",
        auto_start=False,
        poll_updates=False,
        enable_terminal_commands=False,
    )
    commands = []
    bot._run_terminal_command_thread = lambda command: commands.append(command)
    bot._handle_update({"message": {"chat": {"id": 1}, "text": "/terminal: echo blocked"}})
    assert commands == []
