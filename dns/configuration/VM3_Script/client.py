import os
import subprocess
import requests
import time

# The interface for the simulation, change accordingly
SIM_INTERFACE = "enp0s9"

# UI
def clear_screen():
        os.system('cls' if os.name == 'nt' else 'clear')

def setup_dhcp():
        clear_screen()
        
        print(f"Attempting to get IP address via DHCP on {SIM_INTERFACE}...")
        # Use dhclient to request an IP from our DHCP server (VM1)
        subprocess.run(["sudo", "dhclient", "-v", SIM_INTERFACE])
        print("\nDHCP process finished. Current settings:")
        time.sleep(2)
        subprocess.run(["ip", "addr", "show", SIM_INTERFACE])
        input("\nPress Enter to continue...")

def setup_manual():
        clear_screen()
        print("--- Manual IP Configuration ---")
        ip = input("Enter IP Address (e.g., 192.168.100.50): ")
        dns = input("Enter DNS Server IP (e.g., 192.168.100.2): ")

        print("\nApplying settings... (requires sudo)")
        
        commands = [
        # Try to delete any existing default route to prevent conflicts.
        # We add '|| true' so the script doesn't crash if no route exists to delete.
        "sudo ip route del default || true",
        f"sudo ip addr flush dev {SIM_INTERFACE}",
        f"sudo ip addr add {ip}/24 dev {SIM_INTERFACE}",
        f"sudo ip link set dev {SIM_INTERFACE} up",
        "sudo ip route add default via 192.168.100.1",
        f"echo 'nameserver {dns}' | sudo tee /etc/resolv.conf"
        ]
        
        for cmd in commands:
                subprocess.run(cmd, shell=True, check=True)

        print("\nConfiguration applied.")
        input("Press Enter to continue...")

def main_client_loop():
        while True:
                clear_screen()
                print("Enter the domain name to visit (or type 'exit' to quit).")
                domain = input("Domain (e.g., goonshub.local): ")

                if domain.lower() == 'exit':
                        break
                if not domain:
                        continue

                url = f"http://{domain}"
                try:
                        print(f"\nRequesting {url}...")
                        response = requests.get(url, timeout=10)
                        clear_screen()
                        print("Response Received.")
                        print(f"URL: {url}")
                        print(f"Status Code: {response.status_code}")
                        print("Content: \n")
                        print(response.text)
                except requests.exceptions.RequestException as e:
                        print(f"\nAn error occurred: {e}")

                input("\nPress Enter to return to the main menu.")

if __name__ == "__main__":
        clear_screen()
        while True:
                print("--- Network Setup for Simulation ---")
                print("1. Configure network using DHCP (Recommended)")
                print("2. Configure network manually")
                choice = input("Choose an option (1 or 2): ")
                if choice == '1':
                        setup_dhcp()
                        break
                elif choice == '2':
                        setup_manual()
                        break
                else:
                        print("Invalid choice. Please try again.")
                        time.sleep(2)

        main_client_loop()