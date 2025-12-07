import re
import argparse
import ipaddress
from datetime import datetime
from pathlib import Path

import pandas


def parse_header(log_line):
    match_result = re.match(
        r'(?P<date>\d{6})\s+(?P<time>\d{6})\s+(?P<pid>\d+)\s+(?P<level>\w+)\s+(?P<cls>[\w.$]+):\s+(?P<msg>.*)',
        log_line
    )
    if not match_result:
        return None
    return match_result.groupdict()


def parse_ip_port(ip_port_string):
    if not ip_port_string:
        return None, None
    stripped_ip_port = ip_port_string.strip()
    stripped_ip_port = stripped_ip_port.lstrip('/')
    if ':' in stripped_ip_port:
        ip_address_string, port_string = stripped_ip_port.split(':', 1)
        try:
            port_number = int(port_string)
        except Exception:
            port_number = None
        return ip_address_string, port_number
    return stripped_ip_port, None


def ip_to_int(ip_address_string):
    if not ip_address_string:
        return None
    try:
        return int(ipaddress.IPv4Address(ip_address_string))
    except Exception:
        return None


def ip_to_subnet_24(ip_address_string):
    if not ip_address_string:
        return None
    try:
        octet_parts = ip_address_string.split('.')
        if len(octet_parts) != 4:
            return None
        return '.'.join(octet_parts[:3])
    except Exception:
        return None


def parse_message(message_string):
    event_fields = {
        'event_type': None,
        'block_id': None,
        'size': None,
        'src_ip': None,
        'src_port': None,
        'dest_ip': None,
        'dest_port': None,
        'to_ip': None,
        'to_port': None,
        'from_ip': None,
        'path': None,
        'node': None,
        'event_category': None,
        'msg_len': len(message_string) if message_string else 0
    }
    match_result = re.search(
        r'Received block\s+(blk_[\-\d]+)(?:.*?size\s+(\d+))?(?:.*?from\s+(/[\d.]+))?',
        message_string
    )
    if match_result:
        event_fields['event_type'] = 'received_block'
        event_fields['block_id'] = match_result.group(1)
        event_fields['size'] = int(match_result.group(2)) if match_result.group(2) else None
        from_ip_address, _unused_port = parse_ip_port(match_result.group(3))
        event_fields['from_ip'] = from_ip_address
        event_fields['event_category'] = 'data_flow'
        return event_fields
    match_result = re.search(
        r'Receiving block\s+(blk_[\-\d]+)(?:\s+src:\s+(/[\d.:]+))?(?:\s+dest:\s+(/[\d.:]+))?',
        message_string
    )
    if match_result:
        event_fields['event_type'] = 'receiving_block'
        event_fields['block_id'] = match_result.group(1)
        source_ip_address, source_port_number = parse_ip_port(match_result.group(2))
        destination_ip_address, destination_port_number = parse_ip_port(match_result.group(3))
        event_fields['src_ip'] = source_ip_address
        event_fields['src_port'] = source_port_number
        event_fields['dest_ip'] = destination_ip_address
        event_fields['dest_port'] = destination_port_number
        event_fields['event_category'] = 'data_flow'
        return event_fields
    match_result = re.search(r'Served block\s+(blk_[\-\d]+)\s+to\s+(/[\d.]+)', message_string)
    if match_result:
        event_fields['event_type'] = 'served_block'
        event_fields['block_id'] = match_result.group(1)
        to_ip_address, _unused_port = parse_ip_port(match_result.group(2))
        event_fields['to_ip'] = to_ip_address
        event_fields['event_category'] = 'data_flow'
        return event_fields
    match_result = re.search(r'Transmitted block\s+(blk_[\-\d]+)\s+to\s+(/[\d.:]+)', message_string)
    if match_result:
        event_fields['event_type'] = 'transmitted_block'
        event_fields['block_id'] = match_result.group(1)
        to_ip_address, to_port_number = parse_ip_port(match_result.group(2))
        event_fields['to_ip'] = to_ip_address
        event_fields['to_port'] = to_port_number
        event_fields['event_category'] = 'data_flow'
        return event_fields
    match_result = re.search(r'NameSystem\.allocateBlock:\s+([^\n]+?)\.\s+(blk_[\-\d]+)', message_string)
    if match_result:
        event_fields['event_type'] = 'allocate_block'
        event_fields['path'] = match_result.group(1).strip()
        event_fields['block_id'] = match_result.group(2)
        event_fields['event_category'] = 'metadata'
        return event_fields
    match_result = re.search(
        r'NameSystem\.addStoredBlock:\s+.*?:\s+([\d.]+:\d+).*?\s+(blk_[\-\d]+)\s+size\s+(\d+)',
        message_string
    )
    if match_result:
        event_fields['event_type'] = 'add_stored_block'
        event_fields['node'] = match_result.group(1)
        event_fields['block_id'] = match_result.group(2)
        event_fields['size'] = int(match_result.group(3))
        event_fields['event_category'] = 'metadata'
        return event_fields
    match_result = re.search(
        r'ask\s+([\d.]+:\d+)\s+to replicate\s+(blk_[\-\d]+)\s+to datanode(s)\s+(.+)',
        message_string
    )
    if match_result:
        event_fields['event_type'] = 'replicate_block'
        event_fields['node'] = match_result.group(1)
        event_fields['block_id'] = match_result.group(2)
        event_fields['event_category'] = 'metadata'
        return event_fields
    if 'PacketResponder' in message_string and 'terminating' in message_string:
        match_result = re.search(r'PacketResponder.*?block\s+(blk_[\-\d]+)', message_string)
        event_fields['event_type'] = 'packet_responder_terminate'
        event_fields['block_id'] = match_result.group(1) if match_result else None
        event_fields['event_category'] = 'control'
        return event_fields
    event_fields['event_type'] = 'other'
    match_result = re.search(r'\b(blk_[\-\d]+)\b', message_string)
    if match_result:
        event_fields['block_id'] = match_result.group(1)
    event_fields['event_category'] = 'other'
    return event_fields


def parse_log_file(input_file_path):
    records = []
    with open(input_file_path, 'r', encoding='utf-8', errors='ignore') as file_handle:
        for log_line in file_handle:
            log_line = log_line.rstrip('\n')
            if not log_line.strip():
                continue
            header_fields = parse_header(log_line)
            if not header_fields:
                continue
            message_fields = parse_message(header_fields['msg'])
            record = {}
            record.update(header_fields)
            record.update(message_fields)
            records.append(record)
    dataframe = pandas.DataFrame(records)

    def to_timestamp(dataframe_row):
        date_string = dataframe_row['date']
        time_string = dataframe_row['time']
        try:
            year = 2000 + int(date_string[:2])
            month = int(date_string[2:4])
            day = int(date_string[4:6])
            hour = int(time_string[:2])
            minute = int(time_string[2:4])
            second = int(time_string[4:6])
            return datetime(year, month, day, hour, minute, second)
        except Exception:
            return pandas.NaT

    dataframe['timestamp'] = dataframe.apply(to_timestamp, axis=1)
    dataframe['hour'] = dataframe['timestamp'].dt.hour
    dataframe['minute'] = dataframe['timestamp'].dt.minute
    dataframe['second'] = dataframe['timestamp'].dt.second

    for column_name in ['pid']:
        dataframe[column_name] = pandas.to_numeric(dataframe[column_name], errors='coerce')
    for column_name in [
        'size', 'src_port', 'dest_port', 'to_port', 'msg_len', 'hour', 'minute', 'second'
    ]:
        dataframe[column_name] = pandas.to_numeric(dataframe[column_name], errors='coerce')

    dataframe['src_ip_int'] = dataframe['src_ip'].apply(ip_to_int)
    dataframe['dest_ip_int'] = dataframe['dest_ip'].apply(ip_to_int)
    dataframe['to_ip_int'] = dataframe['to_ip'].apply(ip_to_int)
    dataframe['from_ip_int'] = dataframe['from_ip'].apply(ip_to_int)

    dataframe['src_subnet_24'] = dataframe['src_ip'].apply(ip_to_subnet_24)
    dataframe['dest_subnet_24'] = dataframe['dest_ip'].apply(ip_to_subnet_24)
    dataframe['to_subnet_24'] = dataframe['to_ip'].apply(ip_to_subnet_24)
    dataframe['from_subnet_24'] = dataframe['from_ip'].apply(ip_to_subnet_24)

    dataframe['is_same_src_dest'] = (dataframe['src_ip'] == dataframe['dest_ip']).astype('int')

    incident_mask = dataframe['level'].str.upper().isin(['ERROR', 'WARN']) & \
                    ~dataframe['event_type'].str.contains('terminate', na=False)
    dataframe['incident'] = incident_mask.astype('int')

    dataframe = dataframe.drop_duplicates(
        subset=['timestamp', 'pid', 'cls', 'event_type', 'block_id', 'msg_len']
    )
    return dataframe


def run(input_path, output_dataframe_path=None):
    dataframe = parse_log_file(input_path)

    if output_dataframe_path:
        output_path = Path(output_dataframe_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix.lower() != '.csv':
            output_path = output_path.with_suffix('.csv')
        dataframe.to_csv(output_path, index=False)

    return dataframe



if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('--input', required=True)
    argument_parser.add_argument('--out_df', default=None)
    parsed_arguments = argument_parser.parse_args()
    run(
        input_path=parsed_arguments.input,
        output_dataframe_path=parsed_arguments.out_df
    )
