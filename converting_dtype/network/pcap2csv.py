import pyshark
import csv
from collections import defaultdict


def extract_custom_fields(packet, flow_data):
    """Extract custom fields from a packet."""
    try:
        # Extract common fields
        frame_time_delta = float(packet.frame_info.time_delta)  # Time delta in seconds
        flow_data['Flow Duration'] += frame_time_delta * 1e6  # Convert to microseconds

        # Extract packet direction (forward or backward)
        if hasattr(packet, 'ip') and hasattr(packet, 'tcp'):
            if 'src' in packet.ip.field_names and 'dst' in packet.ip.field_names:
                if packet.ip.src < packet.ip.dst:
                    flow_data['Total Fwd Packets'] += 1
                    flow_data['Fwd Packets'] += int(packet.tcp.len or 0)
                else:
                    flow_data['Bwd Packets'] += int(packet.tcp.len or 0)

        # SYN/FIN flags
        if hasattr(packet, 'tcp'):
            flow_data['SYN Flag Count'] += int(packet.tcp.flags_syn == '1')
            flow_data['FIN Flag Count'] += int(packet.tcp.flags_fin == '1')

        # Update timestamps for Active Mean and Idle Mean
        timestamp = float(packet.frame_info.time_epoch)
        flow_data['timestamps'].append(timestamp)

    except AttributeError:
        pass


def finalize_features(flow_data):
    """Calculate derived features like Active Mean and Idle Mean."""
    timestamps = flow_data['timestamps']

    if len(timestamps) > 1:
        time_differences = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]

        # Active Mean: Time differences less than 1 second
        active_times = [t for t in time_differences if t < 1]
        flow_data['Active Mean'] = sum(active_times) / len(active_times) if active_times else 0

        # Idle Mean: Time differences greater than or equal to 1 second
        idle_times = [t for t in time_differences if t >= 1]
        flow_data['Idle Mean'] = sum(idle_times) / len(idle_times) if idle_times else 0
    else:
        flow_data['Active Mean'] = 0
        flow_data['Idle Mean'] = 0

    # Remove timestamps (no need to output)
    del flow_data['timestamps']


def pcap_to_csv(input_file, output_file):
    """Convert PCAP file to CSV with custom fields."""
    # Open the PCAP file
    capture = pyshark.FileCapture(input_file)

    headers = [
        'Flow Duration', 'Total Fwd Packets', 'Fwd Packets', 'Bwd Packets',
        'Flow IAT', 'SYN Flag Count', 'FIN Flag Count', 'Active Mean', 'Idle Mean', 'Label'
    ]

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()

        # Flow data storage
        flow_data = defaultdict(lambda: 0)
        flow_data['timestamps'] = []

        for packet_number, packet in enumerate(capture, start=1):
            extract_custom_fields(packet, flow_data)

            if packet_number % 100 == 0:
                print(f"Processed {packet_number} packets")

        # Finalize and write features
        finalize_features(flow_data)
        flow_data['Label'] = 'BENIGN'  # Update this dynamically if needed
        writer.writerow(flow_data)

    print(f"Conversion complete. Output saved to {output_file}")


if __name__ == "__main__":
    # Hardcoded paths
    input_file = "test.pcap"  # Replace with your PCAP file path
    output_file = "output_pcap.csv"  # Replace with your desired output CSV path

    pcap_to_csv(input_file, output_file)
