#### **1. Configure Internal Network:** 

1. Edit the interfaces file to add the static IP for simulation network (`enp0s9`).

        sudo nano /etc/network/interfaces

2. Add the configuration below:

        # The Internal Network interface (for simulation)
        auto enp0s9
        iface enp0s9 inet static
                address 192.168.100.4
                netmask 255.255.255.0
                gateway 192.168.100.1
                dns-nameservers 127.0.0.1

3. Save the file and activate the new interface:

        sudo ifup enp0s9

4. Enable IP Forwarding: 

        echo "net.ipv4.ip_forward=1" | sudo tee /etc/sysctl.d/99-forwarding.conf && sudo sysctl -p

**Apt Installs:** `sudo apt install python3-flask python3-requests iptables-persistent -y`

#### **2. Configure Firewall:** 

#### First Time Setup: 

1.  Create Firewall Shell Script: `sudo nano /usr/local/bin/setup_firewall.sh`

        #!/bin/bash
        sudo iptables -A INPUT -i lo -j ACCEPT
        sudo iptables -A INPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
        sudo iptables -A INPUT -p tcp --dport 22 -j ACCEPT
        sudo iptables -A INPUT -m iprange --src-range 192.168.100.200-192.168.100.210 -j DROP
        sudo iptables -A INPUT -i enp0s9 -p tcp --dport 80 -j ACCEPT
        sudo iptables -A FORWARD -i enp0s9 -d 192.168.100.3 -p tcp --dport 8080 -j ACCEPT

2.  Run Script:

        sudo chmod +x /usr/local/bin/setup_firewall.sh
        sudo /usr/local/bin/setup_firewall.sh

3.  Save Firewall Rules:

        sudo netfilter-persistent save

4.  Reverse Proxy Python Program: `nano proxy_app.py`

        check directory /VM4_script for client.py

5.  Run the Proxy Program: 

        python3 proxy_app.py


