# Import necessary libraries
import pyshark
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# Load the trained model, scaler, and encoders (you can load these globally)
model = load_model('intrusion_detection_model.h5')

with open('label_encoders.pkl', 'rb') as f:
    le = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

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
# Define a context to hold recent packet info
recent_connections = []


def update_recent_connections(packet):
    recent_connections.append(packet)
    if len(recent_connections) > 1000:
        recent_connections.pop(0)

def extract_features(packet):
    try:
        # Initialize a dictionary with all required feature names and default values
        features = {
            'duration': 0,
            'protocol_type': 'unknown',
            'service': 'unknown',
            'flag': 'unknown',
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
            'is_host_login': 0,
            'is_guest_login': 0,
            'count': 0,
            'srv_count': 0,
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
            'last_flag': 0,
        }

        # Extract specific fields from the packet
        features['protocol_type'] = packet.transport_layer.lower() if hasattr(packet, 'transport_layer') else 'unknown'
        if hasattr(packet, 'tcp'):
            features['src_bytes'] = int(packet.tcp.len) if hasattr(packet.tcp, 'len') else 0
        elif hasattr(packet, 'udp'):
            features['src_bytes'] = int(packet.udp.length) if hasattr(packet.udp, 'length') else 0

        # Add more packet-based logic if available
        # (e.g., extract `service`, `flag`, `duration` if possible)

        return features

    except Exception as e:
        print(f"Error processing packet: {e}")
        return None

#def preprocess_features(features_dict):
    # Convert the dictionary to a DataFrame
  #  features_df = pd.DataFrame([features_dict])
    
    # Define the columns in the exact order as in training
 #   feature_columns = [
  #      'protocol_type', 'src_bytes', 'dst_bytes', 'count', 'srv_count', 
   #     'same_srv_rate', 'diff_srv_rate', 'wrong_fragment', 'hot', 
    #    'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 
    #    'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 
    #    'is_host_login', 'is_guest_login', 'serror_rate', 'srv_serror_rate', 
    #   'rerror_rate', 'srv_rerror_rate', 'srv_diff_host_rate', 'dst_host_count', 
    #    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 
    #    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 
    #    'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 
    #    'dst_host_srv_rerror_rate', 'last_flag'
    #]
    
    # Reindex the DataFrame to ensure the correct column order
    #features_df = features_df.reindex(columns=feature_columns, fill_value=0)
    
    # Encode categorical features (e.g., 'protocol_type')
    #features_df['protocol_type'] = le['protocol_type'].transform(features_df['protocol_type'])
    
    #return features_df

#predict attack
def predict_attack(features):
    if features:
        # Convert the features dictionary to a DataFrame
        features_df = pd.DataFrame([features])
        
        # Ensure all feature names expected by the model are present
        expected_features = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 
            'wrong_fragment', 'hot', 'logged_in', 'num_compromised', 'root_shell',
            'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
            'num_access_files', 'is_host_login', 'is_guest_login', 'count', 'srv_count',
            'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
            'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
            'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
            'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
            'dst_host_srv_rerror_rate', 'last_flag'
        ]
        
        for feature in expected_features:
            if feature not in features_df.columns:
                features_df[feature] = 0  # Default value for missing numerical features

        # Map unseen categories to 'other' for categorical features
        categorical_columns = ['protocol_type', 'service', 'flag']
        for column in categorical_columns:
            if column in features_df:
                features_df[column] = features_df[column].apply(
                    lambda x: x if x in le[column].classes_ else 'other'
                )
        
        # Scale the features
        scaled_features = scaler.transform(features_df)
        
        # Predict using the model
        prediction = model.predict(scaled_features)
        predicted_attack_idx = np.argmax(prediction)
        attack_name = le['attack'].inverse_transform([predicted_attack_idx])[0]
        attack_category = categorize_attack(attack_name)
        print(f"Predicted Attack: {attack_name}, Category: {attack_category}")


def main():
    capture = pyshark.LiveCapture(interface='Wi-Fi', tshark_path='C:\\Program Files\\Wireshark\\tshark.exe')

    # Capture only one packet
    capture.sniff(packet_count=1)
    
    # Get the first packet  
    if capture:
        packet = capture[0]
        print(packet)
        if not hasattr(packet, 'transport_layer'):
            print("Non-transport layer packet detected. Skipping...")
        else:
            print(packet)
            features = extract_features(packet)
        if features:
            print("Extracted Features:", features)
            predict_attack(features)


if __name__ == "__main__":
    main()