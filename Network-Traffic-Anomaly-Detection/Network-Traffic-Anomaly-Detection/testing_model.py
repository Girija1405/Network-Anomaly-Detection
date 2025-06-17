import pandas as pd
import numpy as np
import pickle
from tensorflow import keras
import pyshark
# Define the sample data point
capture = pyshark.LiveCapture(interface='Wi-Fi', tshark_path='C:\\Program Files\\Wireshark\\tshark.exe')

# Capture only one packet
capture.sniff(packet_count=1)

# Get the first packet
if capture:
    packet = capture[0]
    print("Captured Packet:", packet)

sample_data= {
    # Sample data for a U2R (User to Root) attack
    'duration': 300,  # Longer duration for attacks that involve privilege escalation
    'protocol_type': 'tcp',  # Protocol type used for exploiting vulnerabilities
    'service': 'ftp',  # Often, U2R attacks target services like FTP or SSH
    'flag': 'SF',  # Standard flag for typical U2R attack scenarios
    'src_bytes': 1200,  # Moderate number of source bytes indicating data transfer during the attack
    'dst_bytes': 500,  # Some data sent back from the victim system
    'wrong_fragment': 0,  # Typically, no fragmentation involved in U2R
    'hot': 0,  # Hot (highly suspicious) ports are not necessarily part of U2R
    'logged_in': 1,  # Attacker might be logged into the victim system
    'num_compromised': 1,  # System might be compromised, especially in U2R attacks
    'root_shell': 1,  # Root shell is usually the goal of a U2R attack
    'su_attempted': 1,  # The attacker likely attempted to gain superuser privileges
    'num_root': 1,  # The attacker gained root access
    'num_file_creations': 3,  # Attackers might create files to exploit the system
    'num_shells': 1,  # A shell may be invoked to gain control of the system
    'num_access_files': 5,  # Access to files, especially critical system files
    'is_host_login': 1,  # Likely a host login scenario for U2R
    'is_guest_login': 0,  # The attacker is not using guest login in this case
    'count': 10,  # A higher count might indicate repeated attempts to exploit vulnerabilities
    'srv_count': 8,  # The attacker might also send a large number of requests to the victim service
    'serror_rate': 0.2,  # Some service errors expected during exploitation attempts
    'srv_serror_rate': 0.3,  # Increased service errors on the server side
    'rerror_rate': 0.1,  # Some remote errors due to the exploitation attempt
    'srv_rerror_rate': 0.2,  # Slightly increased server-side remote errors
    'same_srv_rate': 0.5,  # The attacker might try to access the same service multiple times
    'diff_srv_rate': 0.2,  # Minor difference in the services being targeted
    'srv_diff_host_rate': 0.1,  # Attack might involve different hosts (targeting multiple systems)
    'dst_host_count': 3,  # Attacker might target several destination hosts
    'dst_host_srv_count': 5,  # Multiple service requests sent to the destination hosts
    'dst_host_same_srv_rate': 0.8,  # The same service is often attacked repeatedly
    'dst_host_diff_srv_rate': 0.2,  # Some attempts to attack different services
    'dst_host_same_src_port_rate': 0.9,  # Same source port is used for most requests
    'dst_host_srv_diff_host_rate': 0.1,  # Less likelihood of attacking different hosts
    'dst_host_serror_rate': 0.4,  # Service errors expected from the destination system
    'dst_host_srv_serror_rate': 0.5,  # High service error rate from the destination service
    'dst_host_rerror_rate': 0.0,  # No remote errors in the destination system
    'dst_host_srv_rerror_rate': 0.0,  # No remote service errors in the destination system
    'last_flag': 1  # The last flag is 1 to indicate that the attack has reached its peak (root access achieved)
}



# Log the raw data packet
print("Raw data packet:", sample_data)

# Convert the sample data to a DataFrame
sample_df = pd.DataFrame([sample_data])

# Load the scaler and label encoders
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

categorical_columns = ['protocol_type', 'service', 'flag']
# Encode categorical features
for column in categorical_columns:
    le = label_encoders[column]
    if sample_df[column].iloc[0] in le.classes_:
        sample_df[column] = le.transform(sample_df[column])
    else:
        # Handle unseen categories
        sample_df[column] = -1  # or some other appropriate value

# Prepare features for prediction
X_sample = sample_df.drop(columns=['attack'], errors='ignore')  # 'attack' might not be present
X_sample_scaled = scaler.transform(X_sample)

# Load the trained model
model = keras.models.load_model('intrusion_detection_model.h5')

# Make prediction
predicted_probabilities = model.predict(X_sample_scaled)
predicted_label = np.argmax(predicted_probabilities, axis=1)  # This gives you the encoded attack label

# Decode the predicted class
attack_le = label_encoders['attack']
predicted_attack = attack_le.inverse_transform(predicted_label)[0]  # Get the original attack name

# Define a dictionary to map attack types to their classes
attack_class_mapping = {
    'DoS': ['back', 'land', 'neptune', 'pod', 'smurf', 'teardrop', 'apache2', 'udpstorm', 'processtable', 'worm'],
    'Probe': ['satan', 'ipsweep', 'nmap', 'portsweep', 'mscan', 'saint'],
    'R2L': ['guess_password', 'ftp_write', 'imap', 'phf', 'multihop', 'warezmaster', 'warezclient', 'spy', 'xlock', 'xsnoop', 'snmpguess', 'snmpgetattack', 'httptunnel', 'sendmail', 'named'],
    'U2R': ['buffer_overflow', 'loadmodule', 'rootkit', 'perl', 'sqlattack', 'xterm', 'ps']
}

# Function to categorize attack
def categorize_attack(predicted_attack):
    for attack_class, attack_types in attack_class_mapping.items():
        if predicted_attack in attack_types:
            return attack_class
    return 'Unknown'

# Categorize the attack
attack_category = categorize_attack(predicted_attack)

# Output the result
print(f"The predicted attack is: '{predicted_attack}'")
print(f"It belongs to the '{attack_category}' category.")