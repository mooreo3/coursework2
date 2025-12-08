import pandas as pd

from preprocess import (
    parse_header,
    parse_message,
    ip_to_int,
    ip_to_subnet_24,
)

def test_parse_header_valid_line():
    line = "240101 123045 123 INFO org.apache.hadoop.hdfs.DataNode: Received block blk_-1"
    header = parse_header(line)
    assert header is not None
    assert header["date"] == "240101"
    assert header["time"] == "123045"
    assert header["pid"] == "123"
    assert header["level"] == "INFO"
    assert header["cls"] == "org.apache.hadoop.hdfs.DataNode"
    assert header["msg"].startswith("Received block blk_-1")


def test_parse_header_invalid_line():
    line = "this is not a valid header"
    header = parse_header(line)
    assert header is None


def test_parse_message_received_block():
    msg = "Received block blk_-1 of size 4096 from /10.0.0.1"
    fields = parse_message(msg)
    assert fields["event_type"] == "received_block"
    assert fields["block_id"] == "blk_-1"
    assert fields["size"] == 4096
    assert fields["from_ip"] == "10.0.0.1"
    assert fields["event_category"] == "data_flow"
    assert fields["msg_len"] == len(msg)


def test_parse_message_served_block():
    msg = "Served block blk_1234 to /192.168.1.10"
    fields = parse_message(msg)
    assert fields["event_type"] == "served_block"
    assert fields["block_id"] == "blk_1234"
    assert fields["to_ip"] == "192.168.1.10"
    assert fields["event_category"] == "data_flow"


def test_ip_to_int_valid():
    assert ip_to_int("127.0.0.1") is not None
    assert ip_to_int("10.0.0.1") is not None


def test_ip_to_int_invalid():
    assert ip_to_int(None) is None
    assert ip_to_int("") is None
    assert ip_to_int("not_an_ip") is None


def test_ip_to_subnet_24_valid():
    assert ip_to_subnet_24("10.0.0.1") == "10.0.0"
    assert ip_to_subnet_24("192.168.1.15") == "192.168.1"


def test_ip_to_subnet_24_invalid():
    assert ip_to_subnet_24(None) is None
    assert ip_to_subnet_24("garbage") is None
    assert ip_to_subnet_24("1.2.3") is None
