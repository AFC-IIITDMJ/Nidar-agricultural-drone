import tkinter as tk
from tkinter import ttk
from tkintermapview import TkinterMapView
from mavsdk import System
import asyncio
import threading
import queue
from functools import partial

class DroneTracker:
    def __init__(self, system_address, drone_name, color):
        self.drone = System()
        self.system_address = system_address
        self.drone_name = drone_name
        self.color = color
        self.running = True
        self.connection_state = "Disconnected"
        self.gps_state = "No Fix"
        self.battery_level = "N/A"
        self.flight_mode = "N/A"
        self.position_queue = queue.Queue()
        self.telemetry_queue = queue.Queue()
        self.last_position = None
        self.thread = threading.Thread(target=self.run_async_tasks, daemon=True)
        self.thread.start()

    async def drone_main(self):
        """Main async coroutine that manages all drone operations"""
        await self.connect_to_drone()
        
        # Run all telemetry tasks concurrently
        await asyncio.gather(
            self.position_updater(),
            self.health_updater(),
            self.battery_updater(),
            self.flight_mode_updater()
        )

    async def connect_to_drone(self):
        try:
            self.update_status("connection", "Connecting...")
            
            await self.drone.connect(system_address=self.system_address)
            
            async for state in self.drone.core.connection_state():
                if state.is_connected:
                    self.update_status("connection", "Connected")
                    break
                
        except Exception as e:
            self.update_status("connection", f"Error: {str(e)}")
            raise

    async def position_updater(self):
        """Continuously updates position"""
        async for position in self.drone.telemetry.position():
            if not self.running:
                break
            if position.latitude_deg and position.longitude_deg:
                self.position_queue.put(("position", (position.latitude_deg, position.longitude_deg)))

    async def health_updater(self):
        """Updates GPS health status"""
        async for health in self.drone.telemetry.health():
            if not self.running:
                break
            status = "GPS Fix" if health.is_global_position_ok else "No Fix"
            self.update_status("gps", status)

    async def battery_updater(self):
        """Updates battery status"""
        async for battery in self.drone.telemetry.battery():
            if not self.running:
                break
            percent = battery.remaining_percent * 100
            self.update_status("battery", f"{percent:.1f}%")

    async def flight_mode_updater(self):
        """Updates current flight mode"""
        async for mode in self.drone.telemetry.flight_mode():
            if not self.running:
                break
            self.update_status("mode", str(mode).split('.')[-1])

    def update_status(self, key, value):
        """Thread-safe status update"""
        self.telemetry_queue.put((key, value))

    def run_async_tasks(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(self.drone_main())
        finally:
            loop.close()

    def stop(self):
        self.running = False

class DroneMapApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Drone Locations")
        self.root.geometry("1200x800")

        # Setup UI
        self.setup_ui()
        
        # Create two drone trackers with different colors
        self.drone1 = DroneTracker("serial://COM5:57600", "Drone 1", "blue")
        self.drone2 = DroneTracker("serial://COM6:57600", "Drone 2", "red")
        
        # Markers for both drones
        self.marker1 = None
        self.marker2 = None
        self.path1 = None
        self.path2 = None
        
        # Start update loops
        self.update_map_loop()
        self.update_status_loop()

    def setup_ui(self):
        # Main container
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Map widget (single map for both drones)
        self.map_widget = TkinterMapView(main_frame, width=800, height=700)
        self.map_widget.pack(side="left", fill="both", expand=True)
        self.map_widget.set_zoom(16)
        self.map_widget.set_position(23.1766, 80.0261)
        
        # Status panels frame
        status_frame = tk.Frame(main_frame, width=300, bg="#f0f0f0")
        status_frame.pack(side="right", fill="y", padx=(10, 0))
        status_frame.pack_propagate(False)
        
        # Drone 1 status panel
        self.setup_drone_status_panel(status_frame, "Drone 1", "blue")
        
        # Separator
        tk.Frame(status_frame, height=2, bg="gray").pack(fill="x", pady=20)
        
        # Drone 2 status panel
        self.setup_drone_status_panel(status_frame, "Drone 2", "red")

    def setup_drone_status_panel(self, parent, drone_name, color):
        # Frame for this drone's status
        drone_frame = tk.Frame(parent, bg="#f0f0f0")
        drone_frame.pack(fill="x", padx=10, pady=5)
        
        # Drone name label with color
        tk.Label(drone_frame, text=drone_name, font=("Arial", 12, "bold"), 
                fg=color, bg="#f0f0f0").pack(anchor="w", pady=(0, 10))
        
        # Status widgets
        if drone_name == "Drone 1":
            self.connection_label1 = self.create_status_row(drone_frame, "Connection:", "Disconnected", "red")
            self.gps_label1 = self.create_status_row(drone_frame, "GPS Status:", "No Fix", "red")
            self.battery_label1 = self.create_status_row(drone_frame, "Battery:", "N/A", "black")
            self.mode_label1 = self.create_status_row(drone_frame, "Flight Mode:", "N/A", "black")
            self.battery_bar1 = self.create_battery_bar(drone_frame)
            self.connect_btn1 = tk.Button(drone_frame, text="Connect", command=partial(self.toggle_connection, 1))
            self.connect_btn1.pack(pady=10, ipadx=10, ipady=5)
        else:
            self.connection_label2 = self.create_status_row(drone_frame, "Connection:", "Disconnected", "red")
            self.gps_label2 = self.create_status_row(drone_frame, "GPS Status:", "No Fix", "red")
            self.battery_label2 = self.create_status_row(drone_frame, "Battery:", "N/A", "black")
            self.mode_label2 = self.create_status_row(drone_frame, "Flight Mode:", "N/A", "black")
            self.battery_bar2 = self.create_battery_bar(drone_frame)
            self.connect_btn2 = tk.Button(drone_frame, text="Connect", command=partial(self.toggle_connection, 2))
            self.connect_btn2.pack(pady=10, ipadx=10, ipady=5)

    def create_status_row(self, parent, label_text, value_text, color):
        frame = tk.Frame(parent, bg="#f0f0f0")
        frame.pack(fill="x", padx=5, pady=2)
        tk.Label(frame, text=label_text, bg="#f0f0f0").pack(side="left")
        value_label = tk.Label(frame, text=value_text, fg=color, bg="#f0f0f0")
        value_label.pack(side="right")
        return value_label

    def create_battery_bar(self, parent):
        style = ttk.Style()
        style.configure("green.Horizontal.TProgressbar", foreground="green", background="green")
        style.configure("yellow.Horizontal.TProgressbar", foreground="yellow", background="yellow")
        style.configure("red.Horizontal.TProgressbar", foreground="red", background="red")
        
        battery_bar = ttk.Progressbar(parent, orient="horizontal", length=250, 
                                    style="green.Horizontal.TProgressbar")
        battery_bar.pack(pady=(0, 10), fill="x")
        return battery_bar

    def toggle_connection(self, drone_num):
        """Placeholder for connection toggle logic"""
        if drone_num == 1:
            current = self.connection_label1["text"]
            if "Connected" in current:
                self.connection_label1.config(text="Disconnected", fg="red")
            elif "Disconnected" in current:
                self.connection_label1.config(text="Connecting...", fg="orange")
        else:
            current = self.connection_label2["text"]
            if "Connected" in current:
                self.connection_label2.config(text="Disconnected", fg="red")
            elif "Disconnected" in current:
                self.connection_label2.config(text="Connecting...", fg="orange")

    def update_map_loop(self):
        """Process position updates in the main thread"""
        # Update Drone 1 position
        try:
            while not self.drone1.position_queue.empty():
                msg_type, data = self.drone1.position_queue.get_nowait()
                if msg_type == "position":
                    lat, lon = data
                    self.update_map_position(1, lat, lon)
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Drone 1 map update error: {e}")
        
        # Update Drone 2 position
        try:
            while not self.drone2.position_queue.empty():
                msg_type, data = self.drone2.position_queue.get_nowait()
                if msg_type == "position":
                    lat, lon = data
                    self.update_map_position(2, lat, lon)
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Drone 2 map update error: {e}")
        
        self.root.after(100, self.update_map_loop)

    def update_map_position(self, drone_num, lat, lon):
        """Update map with new position for specific drone"""
        if drone_num == 1:
            # Update Drone 1 position
            if self.marker1:
                self.map_widget.delete(self.marker1)
            self.marker1 = self.map_widget.set_marker(lat, lon, text="Drone 1", 
                                                   text_color="blue", marker_color_circle="blue",
                                                   marker_color_outside="blue")
            
            if self.drone1.last_position:
                if self.path1:
                    self.map_widget.delete(self.path1)
                self.path1 = self.map_widget.set_path([self.drone1.last_position, (lat, lon)], color="blue")
            self.drone1.last_position = (lat, lon)
        else:
            # Update Drone 2 position
            if self.marker2:
                self.map_widget.delete(self.marker2)
            self.marker2 = self.map_widget.set_marker(lat, lon, text="Drone 2", 
                                                   text_color="red", marker_color_circle="red",
                                                   marker_color_outside="red")
            
            if self.drone2.last_position:
                if self.path2:
                    self.map_widget.delete(self.path2)
                self.path2 = self.map_widget.set_path([self.drone2.last_position, (lat, lon)], color="red")
            self.drone2.last_position = (lat, lon)

    def update_status_loop(self):
        """Process telemetry updates in the main thread"""
        # Update Drone 1 status
        try:
            while not self.drone1.telemetry_queue.empty():
                key, value = self.drone1.telemetry_queue.get_nowait()
                self.process_status_update(1, key, value)
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Drone 1 status update error: {e}")
        
        # Update Drone 2 status
        try:
            while not self.drone2.telemetry_queue.empty():
                key, value = self.drone2.telemetry_queue.get_nowait()
                self.process_status_update(2, key, value)
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Drone 2 status update error: {e}")
        
        self.root.after(100, self.update_status_loop)

    def process_status_update(self, drone_num, key, value):
        """Update specific status field for specific drone"""
        if drone_num == 1:
            if key == "connection":
                self.drone1.connection_state = value
                color = "green" if value == "Connected" else "orange" if value == "Connecting..." else "red"
                self.connection_label1.config(text=value, fg=color)
            elif key == "gps":
                self.drone1.gps_state = value
                color = "green" if value == "GPS Fix" else "red"
                self.gps_label1.config(text=value, fg=color)
            elif key == "battery":
                self.drone1.battery_level = value
                self.battery_label1.config(text=value)
                try:
                    percent = float(value.strip('%'))
                    self.battery_bar1['value'] = percent
                    if percent > 50:
                        self.battery_bar1.config(style="green.Horizontal.TProgressbar")
                    elif percent > 20:
                        self.battery_bar1.config(style="yellow.Horizontal.TProgressbar")
                    else:
                        self.battery_bar1.config(style="red.Horizontal.TProgressbar")
                except ValueError:
                    pass
            elif key == "mode":
                self.drone1.flight_mode = value
                self.mode_label1.config(text=value)
        else:
            if key == "connection":
                self.drone2.connection_state = value
                color = "green" if value == "Connected" else "orange" if value == "Connecting..." else "red"
                self.connection_label2.config(text=value, fg=color)
            elif key == "gps":
                self.drone2.gps_state = value
                color = "green" if value == "GPS Fix" else "red"
                self.gps_label2.config(text=value, fg=color)
            elif key == "battery":
                self.drone2.battery_level = value
                self.battery_label2.config(text=value)
                try:
                    percent = float(value.strip('%'))
                    self.battery_bar2['value'] = percent
                    if percent > 50:
                        self.battery_bar2.config(style="green.Horizontal.TProgressbar")
                    elif percent > 20:
                        self.battery_bar2.config(style="yellow.Horizontal.TProgressbar")
                    else:
                        self.battery_bar2.config(style="red.Horizontal.TProgressbar")
                except ValueError:
                    pass
            elif key == "mode":
                self.drone2.flight_mode = value
                self.mode_label2.config(text=value)

    def on_closing(self):
        """Cleanup on window close"""
        self.drone1.stop()
        self.drone2.stop()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = DroneMapApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()