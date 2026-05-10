from dotenv import load_dotenv
import os

load_dotenv()
detector = os.getenv('detector')
denoiser = os.getenv('denoiser')
super_detector = os.getenv('super_detector')
super_denoiser = os.getenv('super_denoiser')
path_1_3_1 = os.getenv('1_3_1_path')
path_1_3_2 = os.getenv('1_3_2_path')
path_1_3_3 = os.getenv('1_3_3_path')
