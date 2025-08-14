**Apt Installs:** `sudo apt install python3-requests python3-pip dhcpcd5 -y`

**Client Program:** `nano client.py`

        check directory /VM3_script for client.py

**Restart DHCP**:

        sudo dhclient -r [SIM_INTERFACE]

**Run the Client Program:** `python3 client.py`
