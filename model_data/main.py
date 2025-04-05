from Detector import *
import os

def main():
    # To use the device's local camera, set `use_local_camera` to True.
    # To use a video stream from a URL, set it to False.
    use_local_camera = True

    # When using a URL, set the appropriate server address.
    # This should be a server adress url where the live video stream is being broadcasted not a yt URL.
    # Example: "http://192.168.1.1:8080/video"
    server_address = "http://192.168.1.1/" if not use_local_camera else 0

    # configPath = "\\Your\\ssd_mobilenet\\file\\path"
    # modelPath = "\\Your\\frozen_inference\\file\\path"
    # classesPath = "\\Your\\coco.names\\file\\path"
    # "http://192.168.1.1/"
    configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
    modelPath = "frozen_inference_graph.pb"
    classesPath = "coco.names"
    focalLength = 367  # or whatever the correct value is for your setup

    detector = Detector(server_address, configPath, modelPath, classesPath, focalLength, use_local_camera)
    detector.start_processing()

    input("Press Enter to stop processing...")

    detector.stop_processing()

if __name__ == '__main__':
    main()
