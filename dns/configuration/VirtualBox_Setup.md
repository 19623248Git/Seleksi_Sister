#### **VBox VM Configuration**

Three network adapters on each VM:

  * **Adapter 1 (NAT):** Internet access for apt install commands. 
  * **Adapter 2 (Host-only):** SSH for VM Management. (should be `enp0s8`)
  * **Adapter 3 (Internal Network):** isolated network named `intnet-dns-sandbox` for simulation. (should be `enp0s9`)

#### **VM Roles & IPs (on the Internal Network)**

| VM \#     | Role                     | Internal Network IP (`enp0s9`) |
| :------- | :----------------------- | :----------------------------- |
| **VM 1** | DNS & DHCP Server        | `192.168.100.2`                |
| **VM 2** | HTTP Web Server          | `192.168.100.3`                |
| **VM 3** | Client                   | DHCP / Manual                  |
| **VM 4** | Reverse Proxy & Firewall | `192.168.100.4`                |
| **Domain Name:** `goonshub.local` |                                |

#### One-Time VirtualBox Configuration

1.  Configure VM Adapters: For each of the four VMs (while powered off), go to **Settings** -\> **Network** and configure the three adapters exactly as described in the network design above.