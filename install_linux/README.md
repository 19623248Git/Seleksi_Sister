# Dualboot NixOS Minimal Installation in Windows 11

![NixOShaha.jpeg](/install_linux/img/NixOShaha.jpeg)

## Flash ISO
1. Download the <a href="https://nixos.org/download/#nixos-iso">ISO file</a>, select the **minimal installation**.
2. Download & Install <a href="https://etcher.balena.io/#download-etcher">balenaEtcher</a>.
3. Run balenaEtcher and configure as follows:
   - Click `Flash From File` and pick the `.iso` NixOS minimal install file.
   - Click `Select Target` and pick your USB target.
   - Click `Flash!`

## Disk Partitioning (Windows 11)
1. Open Disk Management, and search for the C: drive (filesystem type: NTFS)
2. Right click the C: drive and click `shrink volume`, wait until querying is done
3. Input the desired amount to be shrinked (1 GB -> 1024 MB)
4. You should now see an unallocated partition

## Initial setups to enter Installation
1. Plug your USB in with the flashed image
2. Open BIOS mode, in windows 11, go to `settings > system > recovery`, pick `Advanced Setup` and click `Restart now`
3. The computer will restarted to safe mode, click `Troubleshoot > Advanced options > UEFI Firmware Settings > restart`
4. Configure the boot order so the `USB UEFI` has more priority than `Windows Boot Manager`
5. MAKE SURE SECURE BOOT IS OFF (CHECK FOR RESPECTIVE BIOS PROVIDERS) OR YOU WON'T BE ABLE TO BOOT INTO THE USB DRIVE
6. `Save the configuration` and exit the BIOS menu, now it should be booted to the USB install

## NixOS Installation
1. When prompted, select the `Linux LTS` installation to begin
2. Before we begin, elevate user by typing in `sudo -s` and ensure wifi connection by doing `ping 8.8.8.8`. Next, type in the command `lsblk`, it will display your disk and its partitions, for my case it was `dev/sda` indicating the boot USB, and `/dev/nvme0n1` indicating windows with its partitions `/dev/nvme0n1px` (x is a number).
3. We are going to create the partitioning with a GPT table using `gdisk`. Start `gdisk` with the comment below:
```bash
gdisk /dev/sdX
```
4. In `gdisk` create two partitions: `swap partition` and `root partition`. To create a new partition input the command `n`.

- `swap partition`: default values for `partition number` and `first sector`. Input `+8G` for `last sector` indicating the size of the partition to be 8 GB. This is done to improve performance. Input hexcode `8200` indicating linux swap
- `root partition`: default values for `partition number`, `first sector`, and `last sector`. This populates the remaining partition size. Input hexcode `8300` indicating linux filesystem

5. Format the partitions as shown below:
```bash
# Format the swap partition
mkswap /dev/nvme0n1px

# Format the root partition with a label for easy access
mkfs.ext4 -L nixos /dev/nvme0n1px
```

6. Mount the Filesystem and the existing Windows EFI partition as shown below:
```bash
# Mount the root partition by its label, more convenient
mount /dev/disk/by-label/nixos /mnt

# Turn on swap in the swap partition
swapon /dev/nvme0n1px

# Create a mount point and mount the EFI partition 
mkdir -p /mnt/boot
mount /dev/nvme0n1px /mnt/boot
```
7. Generate the NixOS configuration file and open with nano to configure:
```bash
nixos-generate-config --root /mnt
nano /mnt/etc/nixos/configuration.nix # Assuming you are a super user, if not then use sudo
```

8. Important configurations to consider for a bare minimum install:
- This guide uses the systemd bootloader, enable the following:
```bash
boot.loader.systemd-boot.enable = true;
boot.loader.efi.canTouchEfiVariables = true;
boot.loader.efi.efiSysMountPoint = "/boot";
```

- Set timezone as follows:
```bash
time.timeZone = "Asia/Jakarta";
```

- Enable NetworkManager for easy connections, recommended and used by a lot of distros:
```bash
networking.networkmanager.enable = true;
```

- Add a user for your system:
```bash
users.users.[YOUR_USERNAME] = {
      isNormalUser = true;
      extraGroups = [ "wheel" ]; # Enable 'sudo' for the user.
      packages = with pkgs; [
        tree
      ];
      password = [YOUR_PASSWORD]; # Not recommended but it works!
    };
```

- Enable sound:
```bash
services.pipewire = {
      enable = true;
      alsa.enable = true;
      pulse.enable = true;
      jack.enable = true;
    };
```

- Install additional packages according to your requirement, for example:
```bash
nixpkgs.config.allowUnfree = true;  # This is for vscode
    environment.systemPackages = with pkgs; [
      vim
      wget
      git
      firefox
      vscode # Requires unfree to be allowed
      gedit
      neofetch
      htop
      wineWowPackages.stable
      winetricks
      polkit_gnome

      # core graphical utilities
      sway # Windows Tiling Manager and uses Wayland
      swaylock
      swayidle
      wofi
      waybar
      alacritty
      wl-clipboard
    ];
```

9. After configuring `/etc/nixos/configuration.nix`, build the configuration using the command below:
```bash
nixos-install
```

10. After installation is successful, `reboot` the system and unplug the USB as the computer restarts (Don't unplug when it's ending its processes). `Systemd-boot` menu will now appear, giving the option to boot into NixOS or Windows.

11. Congratulations, the installation is complete!!!

<br>

# Detail Pengerjaan

### OS: (4 poin) NixOS tanpa menggunakan installer

### Video Demo: Click the <a href="https://drive.google.com/drive/folders/1HKcEf9uJE1ukBdUgpTKsqOSRiff2ATsY?usp=sharing">link!</a>


note: Ada beberapa video demo yang direkam terpisah.

| Bonus | Bukti |
| :--- | :--- |
| **Entertainment** |
| (1 poin) Menginstal dan memainkan game yang tidak memiliki native support terhadap Linux | [video link](https://drive.google.com/file/d/1mT66yCV-RNHfVJ0LAp39TyfZxm-9umTy/view?usp=sharing) |
| (1 poin) Melakukan instalasi di hardware fisik, entah berupa komputer, laptop, atau bahkan removable drive | [screenshot link](https://drive.google.com/file/d/1q2w5-CC3kPhziWQcLoC4olBjLoCp-qY9/view?usp=sharing) |
| (1 poin) Mengemas kembali hasil instalasi ke dalam sebuah .iso yang dapat digunakan untuk menginstalasi "custom distro" Anda di perangkat lain | [iso file link](https://drive.google.com/file/d/1kdDzoK9-TP0qp58nC6_3H3BahCwJQKFw/view?usp=sharing) |
| (0.5 poin) Memasang graphical text editor dan web browser | [screenshot link](https://drive.google.com/file/d/1oeCXfHfyOIs93IGL0Wu6RMAllBXVmCmG/view?usp=sharing) |
| (0.5 poin) Menginstal wine untuk menjalankan program Windows, lalu menginstal dan menjalankan LINE for PC di atasnya | [screenshot link](https://drive.google.com/file/d/118REUOsP2fs6PzgJCux5Vv8lBwsPF2Mm/view?usp=sharing) |
| (0.5 poin) Menggunakan tiling window manager atau OpenBox | [screenshot link](https://drive.google.com/file/d/1ZHEkp-cduYRABu4eslIa5CGhD4zaPkP0/view?usp=sharing) |
| (1 poin) Menggunakan Wayland | [screenshot link](https://drive.google.com/file/d/1ZHEkp-cduYRABu4eslIa5CGhD4zaPkP0/view?usp=sharing) |


