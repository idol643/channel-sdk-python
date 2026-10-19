"""Media batching integration tests.

Covered behavior:
1. 5 consecutive images -> 1 dispatch with batched_sources length == 5
2. img -> img -> text -> img -> 3 dispatches in order: media_batch(2), text, media(1)
3. img -> file -> 2 dispatches (incompatible kinds)
4. enabled=False -> no batching, 5 dispatches
5. merge_text=True: img -> img -> text -> 1 dispatch, carrier is the text, batched_sources == [img, img, text]
6. merge_text=False (default): same sequence still yields 2 dispatches (image batch, then text)
7. merge_text=True with nothing pending: text dispatches normally, nothing is swallowed
8. merge_text=True is per chat: a text in another chat neither absorbs nor is absorbed
"""

import asyncio
import time

import pytest

from lark_channel.channel import (
    ChatQueueConfig,
    DedupConfig,
    MediaBatchConfig,
    TextBatchConfig,
)
from lark_channel.channel.safety import SafetyPipeline
from lark_channel.channel.types import (
    Conversation,
    FileContent,
    Identity,
    ImageContent,
    InboundMessage,
    TextContent,
)


def _img(mid: str, *, chat_id: str = "c1") -> InboundMessage:
    return InboundMessage(
        id=mid,
        create_time=int(time.time() * 1000),
        conversation=Conversation(chat_id=chat_id, chat_type="p2p"),
        sender=Identity(open_id="ou_user"),
        content=ImageContent(image_key=f"img_{mid}"),
        raw_content_type="image",
    )


def _file(mid: str, *, chat_id: str = "c1") -> InboundMessage:
    return InboundMessage(
        id=mid,
        create_time=int(time.time() * 1000),
        conversation=Conversation(chat_id=chat_id, chat_type="p2p"),
        sender=Identity(open_id="ou_user"),
        content=FileContent(file_key=f"f_{mid}"),
        raw_content_type="file",
    )


def _text(mid: str, *, chat_id: str = "c1") -> InboundMessage:
    return InboundMessage(
        id=mid,
        create_time=int(time.time() * 1000),
        conversation=Conversation(chat_id=chat_id, chat_type="p2p"),
        sender=Identity(open_id="ou_user"),
        content=TextContent(text=f"t-{mid}"),
        raw_content_type="text",
    )


@pytest.mark.asyncio
async def test_five_consecutive_images_merge_to_one_dispatch():
    delivered = []

    async def on_message(m):
        delivered.append(m)

    loop = asyncio.get_running_loop()
    pipe = SafetyPipeline(
        loop=loop,
        on_message=on_message,
        media_batch_config=MediaBatchConfig(enabled=True, delay_ms=50, max_items=9),
        # disable text batch + chat_queue + dedup so we isolate media batch behavior
        batch_config=TextBatchConfig(delay_ms=0, max_messages=1),
        queue_config=ChatQueueConfig(enabled=False),
        dedup_config=DedupConfig(enabled=False),
    )
    for i in range(5):
        await pipe.push_message(_img(f"m{i}"))
    await asyncio.sleep(0.2)  # > delay_ms
    assert len(delivered) == 1, f"expected 1 merged dispatch, got {len(delivered)}"
    assert delivered[0].batched_sources is not None
    assert len(delivered[0].batched_sources) == 5


@pytest.mark.asyncio
async def test_img_img_text_img_three_dispatches():
    delivered = []

    async def on_message(m):
        delivered.append(m.raw_content_type)

    loop = asyncio.get_running_loop()
    pipe = SafetyPipeline(
        loop=loop,
        on_message=on_message,
        media_batch_config=MediaBatchConfig(enabled=True, delay_ms=50, max_items=9),
        batch_config=TextBatchConfig(delay_ms=0, max_messages=1),
        queue_config=ChatQueueConfig(enabled=False),
        dedup_config=DedupConfig(enabled=False),
    )
    await pipe.push_message(_img("m1"))
    await pipe.push_message(_img("m2"))
    await pipe.push_message(_text("m3"))           # text forces media flush
    await pipe.push_message(_img("m4"))
    await asyncio.sleep(0.2)

    assert delivered == ["image", "text", "image"], delivered


@pytest.mark.asyncio
async def test_image_file_dispatched_separately():
    delivered = []

    async def on_message(m):
        delivered.append(m.raw_content_type)

    loop = asyncio.get_running_loop()
    pipe = SafetyPipeline(
        loop=loop,
        on_message=on_message,
        media_batch_config=MediaBatchConfig(enabled=True, delay_ms=50),
        batch_config=TextBatchConfig(delay_ms=0, max_messages=1),
        queue_config=ChatQueueConfig(enabled=False),
        dedup_config=DedupConfig(enabled=False),
    )
    await pipe.push_message(_img("m1"))
    await pipe.push_message(_file("m2"))   # file != image, force flush
    await asyncio.sleep(0.2)

    assert delivered == ["image", "file"], delivered


@pytest.mark.asyncio
async def test_disabled_no_batching():
    delivered = []

    async def on_message(m):
        delivered.append(m.id)

    loop = asyncio.get_running_loop()
    pipe = SafetyPipeline(
        loop=loop,
        on_message=on_message,
        media_batch_config=MediaBatchConfig(enabled=False),
        batch_config=TextBatchConfig(delay_ms=0, max_messages=1),
        queue_config=ChatQueueConfig(enabled=False),
        dedup_config=DedupConfig(enabled=False),
    )
    for i in range(5):
        await pipe.push_message(_img(f"m{i}"))
    await asyncio.sleep(0.2)
    assert delivered == ["m0", "m1", "m2", "m3", "m4"]


@pytest.mark.asyncio
async def test_merge_text_folds_followup_text_into_media_batch():
    """img -> img -> text with merge_text=True: ONE dispatch, text last."""
    delivered = []

    async def on_message(m):
        delivered.append(m)

    loop = asyncio.get_running_loop()
    pipe = SafetyPipeline(
        loop=loop,
        on_message=on_message,
        media_batch_config=MediaBatchConfig(
            enabled=True, delay_ms=50, max_items=9, merge_text=True
        ),
        batch_config=TextBatchConfig(delay_ms=0, max_messages=1),
        queue_config=ChatQueueConfig(enabled=False),
        dedup_config=DedupConfig(enabled=False),
    )
    await pipe.push_message(_img("m1"))
    await pipe.push_message(_img("m2"))
    await pipe.push_message(_text("m3"))     # folded in, not flushed separately
    await asyncio.sleep(0.2)

    assert len(delivered) == 1, f"expected 1 merged dispatch, got {len(delivered)}"
    merged = delivered[0]
    assert merged.raw_content_type == "text", merged.raw_content_type
    assert getattr(merged.content, "text", "") == "t-m3"
    assert merged.batched_sources is not None
    assert [m.id for m in merged.batched_sources] == ["m1", "m2", "m3"]


@pytest.mark.asyncio
async def test_merge_text_off_keeps_text_separate():
    """Default (merge_text=False) must keep the old two-dispatch behaviour."""
    delivered = []

    async def on_message(m):
        delivered.append(m.raw_content_type)

    loop = asyncio.get_running_loop()
    pipe = SafetyPipeline(
        loop=loop,
        on_message=on_message,
        media_batch_config=MediaBatchConfig(enabled=True, delay_ms=50, max_items=9),
        batch_config=TextBatchConfig(delay_ms=0, max_messages=1),
        queue_config=ChatQueueConfig(enabled=False),
        dedup_config=DedupConfig(enabled=False),
    )
    await pipe.push_message(_img("m1"))
    await pipe.push_message(_img("m2"))
    await pipe.push_message(_text("m3"))
    await asyncio.sleep(0.2)

    assert delivered == ["image", "text"], delivered


@pytest.mark.asyncio
async def test_merge_text_without_pending_media_dispatches_text_alone():
    """Text with nothing pending must not be swallowed by the merge path."""
    delivered = []

    async def on_message(m):
        delivered.append((m.raw_content_type, getattr(m.content, "text", "")))

    loop = asyncio.get_running_loop()
    pipe = SafetyPipeline(
        loop=loop,
        on_message=on_message,
        media_batch_config=MediaBatchConfig(
            enabled=True, delay_ms=50, merge_text=True
        ),
        batch_config=TextBatchConfig(delay_ms=0, max_messages=1),
        queue_config=ChatQueueConfig(enabled=False),
        dedup_config=DedupConfig(enabled=False),
    )
    await pipe.push_message(_text("m1"))
    await pipe.push_message(_text("m2"))
    await asyncio.sleep(0.2)

    assert delivered == [("text", "t-m1"), ("text", "t-m2")], delivered


@pytest.mark.asyncio
async def test_merge_text_is_scoped_to_one_chat():
    """A text in another chat must not absorb, nor be absorbed by, the batch."""
    delivered = []

    async def on_message(m):
        delivered.append((m.conversation.chat_id, m.raw_content_type))

    loop = asyncio.get_running_loop()
    pipe = SafetyPipeline(
        loop=loop,
        on_message=on_message,
        media_batch_config=MediaBatchConfig(
            enabled=True, delay_ms=50, merge_text=True
        ),
        batch_config=TextBatchConfig(delay_ms=0, max_messages=1),
        queue_config=ChatQueueConfig(enabled=False),
        dedup_config=DedupConfig(enabled=False),
    )
    await pipe.push_message(_img("m1", chat_id="c1"))
    await pipe.push_message(_text("m2", chat_id="c2"))   # different chat
    await asyncio.sleep(0.2)

    got = sorted(delivered)
    assert got == [("c1", "image"), ("c2", "text")], got
