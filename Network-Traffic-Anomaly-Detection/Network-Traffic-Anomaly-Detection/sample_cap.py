# Import necessary libraries
import pyshark
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Load the trained model, scaler, and encoders (load these globally)
model = load_model('intrusion_detection_model.h5')

with open('label_encoders.pkl', 'rb') as f:
    le = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Define attack categories
attack_categories = {
    'DoS': ['back', 'land', 'neptune', 'pod', 'smurf', 'teardrop', 'Apache2', 'Udpstorm', 'Processtable', 'Worm'],
    'Probe': ['Satan', 'Ipsweep', 'Nmap', 'Portsweep', 'Mscan', 'Saint'],
    'R2L': ['Guess_Password', 'Ftp_write', 'Imap', 'Phf', 'Multihop', 'Warezmaster', 'Warezclient', 'Spy', 'Xlock', 'Xsnoop', 'Snmpguess', 'Snmpgetattack', 'Httptunnel', 'Sendmail', 'Named'],
    'U2R': ['Buffer_overflow', 'Loadmodule', 'Rootkit', 'Perl', 'Sqlattack', 'Xterm', 'Ps']
}

# Function to categorize attack
def categorize_attack(attack_name):
    for category, attacks in attack_categories.items():
        if attack_name in attacks:
            return category
    return 'Unknown'

# Feature extraction from packet
# Feature extraction from packet
# Feature extraction from packet
def extract_features(packet):
    try:
        # Use 'unknown' for unsupported transport layers and services
        protocol_type = packet.transport_layer.lower() if hasattr(packet, 'transport_layer') and packet.transport_layer else 'unknown'
        
        # Define features with default values
        features = {
            'duration': 0,
            'protocol_type': protocol_type,
            'service': 'dns' if hasattr(packet, 'udp') and packet.udp.dstport == '53' else 'unknown',  # Default to 'unknown' if 'dns' is unrecognized
            'flag': 'SF',  # Default for non-TCP packets
            'src_bytes': int(packet.udp.length) if hasattr(packet, 'udp') and hasattr(packet.udp, 'length') else 0,
            'dst_bytes': 0,
            'count': 1,
            'srv_count': 1,
            'same_srv_rate': 1.0,
            'diff_srv_rate': 0.0,
            'dst_host_srv_count': 1,
            'dst_host_same_srv_rate': 1.0,
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
            'is_host_login': 0,
            'is_guest_login': 0,
            'serror_rate': 0.0,
            'srv_serror_rate': 0.0,
            'rerror_rate': 0.0,
            'srv_rerror_rate': 0.0,
            'srv_diff_host_rate': 0.0,
            'dst_host_count': 1,
            'dst_host_diff_srv_rate': 0.0,
            'dst_host_same_src_port_rate': 1.0,
            'dst_host_srv_diff_host_rate': 0.0,
            'dst_host_serror_rate': 0.0,
            'dst_host_srv_serror_rate': 0.0,
            'dst_host_rerror_rate': 0.0,
            'dst_host_srv_rerror_rate': 0.0,
            'last_flag': 0
        }

        # Handle ICMPv6 or other protocols without service information
        if 'icmpv6' in packet:
            features['protocol_type'] = 'icmp'  # Default to 'icmp' if compatible with the model

        return features
    except AttributeError as e:
        print(f"Error extracting features: {e}")
        return None

# Preprocess features for model prediction
def preprocess_features(features):
    df = pd.DataFrame([features])  # Create a DataFrame from features

    # Check for each categorical feature if it’s known, else set to 'unknown'
    for col, encoder in le.items():
        if col in df.columns:
            df[col] = df[col].apply(lambda x: x if x in encoder.classes_ else 'unknown')
            df[col] = encoder.transform(df[col])
    
    # Scale the data
    scaled_features = scaler.transform(df)
    return scaled_features

# Predict attack
def predict_attack(features):
    if features:
        processed_features = preprocess_features(features)
        prediction = model.predict(processed_features)
        predicted_attack_idx = np.argmax(prediction)
        attack_name = le.inverse_transform([predicted_attack_idx])[0]
        attack_category = categorize_attack(attack_name)
        print(f"Predicted Attack: {attack_name}, Category: {attack_category}")

# Main capture and detection loop
def main():
    capture = pyshark.LiveCapture(interface='Wi-Fi', tshark_path='C:\\Program Files\\Wireshark\\tshark.exe')

    # Capture only one packet
    capture.sniff(packet_count=1)
    
    # Get the first packet
    if capture:
        packet = capture[0]
        print(packet)
        features = extract_features(packet)
        print(features)
        predict_attack(features)    

if __name__ == "__main__":
    main()
