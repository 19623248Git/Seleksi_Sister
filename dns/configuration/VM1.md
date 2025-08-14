#### **1. Configure Internal Network**

1. Edit the interfaces file to add the static IP for simulation network (`enp0s9`).

        sudo nano /etc/network/interfaces

2. Add the configuration below:

        # The Internal Network interface (for simulation)
        auto enp0s9
        iface enp0s9 inet static
                address 192.168.100.2
                netmask 255.255.255.0
                gateway 192.168.100.1
                dns-nameservers 127.0.0.1

3. Save the file and activate the new interface:

        sudo ifup enp0s9

**Apt Installs:** `sudo apt install bind9 dnsutils isc-dhcp-server -y`

#### **2. BIND9 (DNS) & DHCP Server Setup**

#### First Time Setup: 

1.  BIND9: `sudo nano /etc/bind/named.conf.local`

    ```
    zone "goonshub.local" { type master; file "/etc/bind/db.goonshub.local"; };
    ```
2.  Zone File: `sudo nano /etc/bind/db.goonshub.local`

    ```
    $TTL    604800
    @ IN SOA ns1.goonshub.local. root.goonshub.local. ( 2 604800 86400 2419200 604800 )
    @ IN NS ns1.goonshub.local.
    ns1 IN A 192.168.100.2
    @ IN A 192.168.100.4
    www IN A 192.168.100.4
    ```
3.  Restart BIND9: 
        
        sudo systemctl restart bind9

4.  DHCP Configuration: `sudo nano /etc/default/isc-dhcp-server`, set: 

        INTERFACESv4="enp0s9"

5.  DHCP Pool Configuration: `sudo nano /etc/dhcp/dhcpd.conf` add this to the end:
    ```
    subnet 192.168.100.0 netmask 255.255.255.0 {
      range 192.168.100.100 192.168.100.150;
      option routers 192.168.100.1;
      option domain-name-servers 192.168.100.2;
      option domain-name "goonshub.local";
    }
    ```
6.  Restart DHCP: 

        sudo systemctl restart isc-dhcp-server

