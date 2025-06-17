import pandas as pd
import numpy as np
import pickle
from scapy.all import sniff, IP, TCP, UDP
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, StandardScaler
import time
from collections import defaultdict
import threading
import queue

# ----------------------------
# Load the Trained Model and Preprocessing Objects
# ----------------------------

# Load the trained model
model = load_model('intrusion_detection_model.h5')

# Load the scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load the label encoders
with open('label_encoders.pkl', 'rb') as f:
    le = pickle.load(f)

# Define the number of classes (should match the training)
num_classes = model.output_shape[-1]

# ----------------------------
# Initialize Session Management
# ----------------------------

# Initialize a dictionary to hold session data
sessions = defaultdict(lambda: {
    'start_time': None,
    'end_time': None,
    'protocol_type': None,
    'service': None,
    'flag': None,
    'src_bytes': 0,
    'dst_bytes': 0,
    'wrong_fragment': 0,
    'hot': 0,
    'logged_in': 0,
    'num_compromised': 0,
    'root_shell': 0,
    'su_attempted': 0,
    'num_root': 0,
    'num_file_creations': 0,
    'num_shells': 0,
    'num_access_files': 0,
    'count': 0,
    'srv_count': 0,
    'serror_rate': 0.0,
    'srv_serror_rate': 0.0,
    'rerror_rate': 0.0,
    'srv_rerror_rate': 0.0,
    'same_srv_rate': 0.0,
    'diff_srv_rate': 0.0,
    'srv_diff_host_rate': 0.0,
    'dst_host_count': 0,
    'dst_host_srv_count': 0,
    'dst_host_same_srv_rate': 0.0,
    'dst_host_diff_srv_rate': 0.0,
    'dst_host_same_src_port_rate': 0.0,
    'dst_host_srv_diff_host_rate': 0.0,
    'dst_host_serror_rate': 0.0,
    'dst_host_srv_serror_rate': 0.0,
    'dst_host_rerror_rate': 0.0,
    'dst_host_srv_rerror_rate': 0.0,
    'last_flag': None
})

# Define a timeout for session inactivity (e.g., 60 seconds)
SESSION_TIMEOUT = 60

# ----------------------------
# Threading Setup for Session Processing
# ----------------------------

# Create a thread-safe queue to hold completed sessions
session_queue = queue.Queue()

def process_session_thread():
    """
    Thread function to process sessions from the queue.
    """
    while True:
        session_key, session = session_queue.get()
        if session_key is None:
            break  # Exit signal

        process_session(session_key, session)
        session_queue.task_done()

# Start the processing thread
processing_thread = threading.Thread(target=process_session_thread, daemon=True)
processing_thread.start()

# ----------------------------
# Feature Extraction and Prediction
# ----------------------------

def process_session(session_key, session):
    """
    Process a completed session and make a prediction.
    """
    # Calculate session duration
    duration = session['end_time'] - session['start_time'] if session['start_time'] else 0

    # Extract features from the session
    feature_dict = {
        'duration': duration,
        'protocol_type': session['protocol_type'],
        'service': session['service'],
        'flag': session['flag'],
        'src_bytes': session['src_bytes'],
        'dst_bytes': session['dst_bytes'],
        'wrong_fragment': session['wrong_fragment'],
        'hot': session['hot'],
        'logged_in': session['logged_in'],
        'num_compromised': session['num_compromised'],
        'root_shell': session['root_shell'],
        'su_attempted': session['su_attempted'],
        'num_root': session['num_root'],
        'num_file_creations': session['num_file_creations'],
        'num_shells': session['num_shells'],
        'num_access_files': session['num_access_files'],
        'count': session['count'],
        'srv_count': session['srv_count'],
        'serror_rate': session['serror_rate'],
        'srv_serror_rate': session['srv_serror_rate'],
        'rerror_rate': session['rerror_rate'],
        'srv_rerror_rate': session['srv_rerror_rate'],
        'same_srv_rate': session['same_srv_rate'],
        'diff_srv_rate': session['diff_srv_rate'],
        'srv_diff_host_rate': session['srv_diff_host_rate'],
        'dst_host_count': session['dst_host_count'],
        'dst_host_srv_count': session['dst_host_srv_count'],
        'dst_host_same_srv_rate': session['dst_host_same_srv_rate'],
        'dst_host_diff_srv_rate': session['dst_host_diff_srv_rate'],
        'dst_host_same_src_port_rate': session['dst_host_same_src_port_rate'],
        'dst_host_srv_diff_host_rate': session['dst_host_srv_diff_host_rate'],
        'dst_host_serror_rate': session['dst_host_serror_rate'],
        'dst_host_srv_serror_rate': session['dst_host_srv_serror_rate'],
        'dst_host_rerror_rate': session['dst_host_rerror_rate'],
        'dst_host_srv_rerror_rate': session['dst_host_srv_rerror_rate'],
        'last_flag': session['last_flag']
    }

    # Convert to DataFrame
    df_session = pd.DataFrame([feature_dict])

    # Preprocess categorical features using LabelEncoder
    for column in ['protocol_type', 'service', 'flag', 'last_flag']:
        if pd.isna(df_session.at[0, column]):
            df_session.at[0, column] = 'unknown'  # Handle missing categorical data
        try:
        # Retrieve the LabelEncoder for the current column from the `le` dictionary
            label_encoder = le[column]
            df_session[column] = label_encoder.transform([df_session.at[0, column]])[0]
        except ValueError:
        # Handle unseen categories
            df_session[column] = label_encoder.transform(['unknown'])[0]

    # Handle missing numerical values if any
    df_session.fillna(0, inplace=True)

    # Scale the features
    X_session_scaled = scaler.transform(df_session)

    # Make prediction
    prediction = model.predict(X_session_scaled)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = prediction[0][predicted_class]

    # Decode the predicted class
    attack_type = le.inverse_transform([predicted_class])[0]

    print(f"Session: {session_key} | Predicted Attack: {attack_type} | Confidence: {confidence:.4f}")

def sniff_packets(packet):
    """
    Callback function to process each captured packet.
    """
    if IP not in packet:
        return

    ip_layer = packet[IP]
    src_ip = ip_layer.src
    dst_ip = ip_layer.dst
    protocol = ip_layer.proto  # 6 for TCP, 17 for UDP, etc.

    if TCP in packet:
        src_port = packet[TCP].sport
        dst_port = packet[TCP].dport
        flag = packet[TCP].flags
        service = get_service(dst_port)
    elif UDP in packet:
        src_port = packet[UDP].sport
        dst_port = packet[UDP].dport
        flag = 'U'  # Placeholder for UDP
        service = get_service(dst_port)
    else:
        src_port = 0
        dst_port = 0
        flag = 'Other'
        service = 'unknown'

    session_key = f"{src_ip}:{src_port}-{dst_ip}:{dst_port}-{protocol}"

    # Initialize session if first packet
    session = sessions[session_key]
    if not session['start_time']:
        session['start_time'] = packet.time

    session['end_time'] = packet.time
    session['protocol_type'] = 'TCP' if protocol == 6 else 'UDP' if protocol == 17 else 'Other'
    session['service'] = service
    session['flag'] = str(flag)

    # Update numerical features
    session['src_bytes'] += len(packet)
    session['dst_bytes'] += len(packet)  # Placeholder: Track bytes sent vs. received
    session['count'] += 1

    # Update serror_rate and rerror_rate
    if 'S' in str(flag):
        session['serror_rate'] += 1
    if 'R' in str(flag):
        session['rerror_rate'] += 1

    # Example: Update 'logged_in' if certain conditions are met
    # Implement actual logic based on packet analysis
    # session['logged_in'] += detect_logged_in(packet)

    # Example: Update 'srv_count' based on service diversity
    # Implement logic to track unique services if needed

    # Update rates based on counts
    session['serror_rate'] = session['serror_rate'] / session['count'] if session['count'] > 0 else 0
    session['rerror_rate'] = session['rerror_rate'] / session['count'] if session['count'] > 0 else 0

    # Implement additional feature updates as per KDD definitions
    # ...

    # Check for session timeout
    current_time = time.time()
    for key in list(sessions.keys()):
        if sessions[key]['end_time'] and (current_time - sessions[key]['end_time'] > SESSION_TIMEOUT):
            session_queue.put((key, sessions[key]))
            del sessions[key]

def get_service(dst_port):
    """
    Map destination port to service.
    """
    port_service_mapping = {
        21: 'ftp',
        22: 'ssh',
        23: 'telnet',
        25: 'smtp',
        53: 'domain',
        80: 'http',
        110: 'pop3',
        111: 'rpcbind',
        135: 'msrpc',
        139: 'netbios_ns',
        143: 'imap',
        443: 'https',
        445: 'microsoft_ds',
        993: 'imaps',
        995: 'pop3s',
        # Add more ports and services as needed
    }
    return port_service_mapping.get(dst_port, 'unknown')

# ----------------------------
# Starting Packet Sniffing and Processing
# ----------------------------

def main():
    # Start sniffing (replace 'eth0' with your actual network interface)
    # To list available interfaces, uncomment the following lines:
    # from scapy.all import get_if_list
    # print("Available Interfaces:")
    # for iface in get_if_list():
    #     print(iface)
    
    print("Starting live intrusion detection. Press Ctrl+C to stop.")
    try:
        sniff(filter="ip", prn=sniff_packets, iface="Wi-Fi", store=False)
    except KeyboardInterrupt:
        print("Live intrusion detection stopped.")
        # Send exit signal to the processing thread
        session_queue.put((None, None))
        processing_thread.join()

if __name__ == "__main__":
    main()