import json
import threading

from foundations.events import Event, JsonlEventLogger


def test_jsonl_logger_preserves_concurrent_complete_event_lines(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    logger = JsonlEventLogger(path)
    count = 200

    threads = [
        threading.Thread(
            target=logger.log,
            args=(Event(index / 100.0, "test_event", {"index": index}),),
        )
        for index in range(count)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    events = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(events) == count
    assert {event["payload"]["index"] for event in events} == set(range(count))
