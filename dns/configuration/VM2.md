#### **1. Configure Internal Network:** 

1. Edit the interfaces file to add the static IP for simulation network (`enp0s9`).

        sudo nano /etc/network/interfaces

2. Add the configuration below:

        # The Internal Network interface (for simulation)
        auto enp0s9
        iface enp0s9 inet static
                address 192.168.100.3
                netmask 255.255.255.0
                gateway 192.168.100.1
                dns-nameservers 127.0.0.1

3. Save the file and activate the new interface:

        sudo ifup enp0s9

#### **2. Web Page Creation:**

        mkdir web
        cd web
        echo "<h1>Hello my Goons, tonight we STEAL THE MOON.</h1>" > index.html

#### **3. Start the Server (in `web`):**

        python3 -m http.server 8080 &
