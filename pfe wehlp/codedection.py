import cv2
import pandas as pd
from numba import jit
from ultralytics import YOLO
import cvzone
import math
from cvzone.FaceMeshModule import FaceMeshDetector
from inference_sdk import InferenceHTTPClient
import timeit
from math import sqrt
import requests
from googletrans import Translator
import pyttsx3
from gtts import gTTS
from pydub import AudioSegment
import pygame
import io
from gtts import gTTS
import os
from playsound import playsound
from mtranslate import translate

print(cv2.__version__)
# class to speech
def translateandpronoce(language,class_name):
    translation = translate(class_name, to_language=language)
    #print(f"{translation.origin} ({translation.src}) --> {translation.text} ({translation.dest})")
    tts = gTTS(translation, lang=language)
    tts.save("adnane.mp3")

    # Lecture du fichier audio
    playsound("adnane.mp3")
    os.remove("adnane.mp3")
#real width of classs
realwidth={"person":40,"car":220,"crosswalk":120,"bus":220,"motorcycle":72,"pothole":50,"stairs":90,"tree":100,"truck":220,"Auto":220,"Train":100,"pole":2,"cat":5,"dog":10,"door":80, 'green light': 20, 'left turn':20, 'red Light':20, 'yellow Light':20}
def verifierkey(c):
    for key in realwidth:
      if c==key:
        return 1
    return 0
def verifierkeyclasse(c):
    for key in class_list7:
      if c==key:
        return 1
    return 0


def detect(image, models):
    distance=0
    widthpixel=0
    Widthreel=0
    f = 796
    exist=True
    
    indice=1
    dictionnairepredection = {}
    class_name = "" 
    listclassname=["bus","Car","Door"]
    counter=0
    for m in models:
       
       
          results = m.predict(image)
          print(indice)
        
         

        

        
          a = results[0].boxes.data
          px = pd.DataFrame(a).astype("float")
              
          for index, row in px.iterrows():
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])
            
            d = int(row[5])
            class_name = m.class_list[d]
            f = 796
            print(class_name,float(row[4]))
            
            confidence=row[4]
            confidence=str(confidence)
            point1 = (x1, y1)
            point2 = (x2, y1)
            h= x2-x1
            print("h",h)
            exist=verifierkey(class_name.lower())
            if exist==1:
             H=realwidth[class_name.lower()]
             for i in range(H-10,H+10):
              print("H",H)
              print("x1",x1,"x2",x2,"y1",y1,"y2",y2)
              distance=(f*i)/h
              print("distance",distance)

             if distance>0 and distance<320 :
             
              cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
              translateandpronoce("fr",class_name)
              cv2.putText(image,class_name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

          indice=indice+1
              
    
    return image


 
model1 = YOLO('yolov8n-oiv7600cl.pt') 


class_list1={0: 'Accordion', 1: 'Adhesive tape', 2: 'Aircraft', 3: 'Airplane', 4: 'Alarm clock', 5: 'Alpaca', 6: 'Ambulance', 7: 'Animal', 8: 'Ant', 9: 'Antelope', 10: 'Apple', 11: 'Armadillo', 12: 'Artichoke', 13: 'Auto part', 14: 'Axe', 15: 'Backpack', 16: 'Bagel', 17: 'Baked goods', 18: 'Balance beam', 19: 
'Ball', 20: 'Balloon', 21: 'Banana', 22: 'Band-aid', 23: 'Banjo', 24: 'Barge', 25: 'Barrel', 26: 'Baseball bat', 27: 'Baseball glove', 28: 'Bat (Animal)', 29: 'Bathroom accessory', 30: 'Bathroom cabinet', 31: 'Bathtub', 32: 'Beaker', 33: 'Bear', 34: 'Bed', 35: 'Bee', 36: 'Beehive', 37: 'Beer', 38: 'Beetle', 39: 'Bell pepper', 40: 'Belt', 41: 'Bench', 42: 'Bicycle', 43: 'Bicycle helmet', 44: 'Bicycle wheel', 45: 'Bidet', 46: 'Billboard', 47: 'Billiard table', 48: 'Binoculars', 49: 'Bird', 50: 'Blender', 51: 'Blue jay', 52: 'Boat', 53: 'Bomb', 54: 'Book', 55: 'Bookcase', 56: 'Boot', 57: 'Bottle', 58: 'Bottle opener', 59: 'Bow and arrow', 60: 'Bowl', 61: 'Bowling equipment', 62: 'Box', 63: 'Boy', 64: 'Brassiere', 65: 'Bread', 66: 'Briefcase', 67: 'Broccoli', 68: 'Bronze sculpture', 69: 'Brown bear', 70: 'Building', 71: 'Bull', 72: 'Burrito', 73: 'Bus', 74: 'Bust', 75: 'Butterfly', 76: 'Cabbage', 77: 'Cabinetry', 78: 'Cake', 79: 'Cake stand', 80: 'Calculator', 81: 'Camel', 82: 'Camera', 83: 'Can opener', 84: 'Canary', 85: 'Candle', 86: 'Candy', 87: 'Cannon', 
88: 'Canoe', 89: 'Cantaloupe', 90: 'Car', 91: 'Carnivore', 92: 'Carrot', 93: 'Cart', 94: 'Cassette deck', 95: 'Castle', 96: 'Cat', 97: 'Cat furniture', 98: 'Caterpillar', 99: 'Cattle', 100: 'Ceiling fan', 101: 'Cello', 102: 'Centipede', 103: 'Chainsaw', 104: 'Chair', 105: 'Cheese', 106: 'Cheetah', 107: 'Chest of drawers', 108: 'Chicken', 109: 'Chime', 110: 'Chisel', 111: 'Chopsticks', 112: 'Christmas tree', 113: 'Clock', 114: 'Closet', 115: 'Clothing', 116: 'Coat', 117: 'Cocktail', 118: 'Cocktail shaker', 119: 'Coconut', 120: 'Coffee', 121: 'Coffee cup', 122: 'Coffee table', 123: 'Coffeemaker', 124: 'Coin', 125: 'Common fig', 126: 'Common sunflower', 127: 'Computer keyboard', 128: 'Computer monitor', 129: 'Computer mouse', 130: 'Container', 131: 'Convenience store', 132: 'Cookie', 133: 'Cooking spray', 134: 'Corded phone', 135: 'Cosmetics', 136: 'Couch', 137: 'Countertop', 138: 'Cowboy hat', 139: 'Crab', 
140: 'Cream', 141: 'Cricket ball', 142: 'Crocodile', 143: 'Croissant', 144: 'Crown', 145: 'Crutch', 146: 'Cucumber', 147: 'Cupboard', 148: 'Curtain', 149: 'Cutting board', 150: 'Dagger', 151: 'Dairy Product', 152: 'Deer', 153: 'Desk', 154: 'Dessert', 155: 'Diaper', 156: 'Dice', 157: 'Digital clock', 158: 
'Dinosaur', 159: 'Dishwasher', 160: 'Dog', 161: 'Dog bed', 162: 'Doll', 163: 'Dolphin', 164: 'Door', 165: 'Door handle', 166: 'Doughnut', 167: 'Dragonfly', 168: 'Drawer', 169: 'Dress', 170: 'Drill (Tool)', 171: 'Drink', 172: 'Drinking straw', 173: 'Drum', 174: 'Duck', 175: 'Dumbbell', 176: 'Eagle', 177: 'Earrings', 178: 'Egg (Food)', 179: 'Elephant', 180: 'Envelope', 181: 'Eraser', 182: 'Face powder', 183: 'Facial tissue holder', 184: 'Falcon', 185: 'Fashion accessory', 186: 'Fast food', 187: 'Fax', 188: 'Fedora', 189: 'Filing cabinet', 190: 'Fire hydrant', 191: 'Fireplace', 192: 'Fish', 193: 'Flag', 194: 'Flashlight', 195: 'Flower', 196: 'Flowerpot', 197: 'Flute', 198: 'Flying disc', 199: 'Food', 200: 'Food processor', 201: 'Football', 202: 'Football helmet', 203: 'Footwear', 204: 'Fork', 205: 'Fountain', 206: 'Fox', 207: 'French fries', 208: 'French horn', 209: 'Frog', 210: 'Fruit', 211: 'Frying pan', 212: 'Furniture', 213: 'Garden Asparagus', 214: 'Gas stove', 215: 'Giraffe', 216: 'Girl', 217: 'Glasses', 218: 'Glove', 219: 'Goat', 220: 'Goggles', 221: 
'Goldfish', 222: 'Golf ball', 223: 'Golf cart', 224: 'Gondola', 225: 'Goose', 226: 'Grape', 227: 'Grapefruit', 228: 'Grinder', 229: 'Guacamole', 230: 'Guitar', 231: 'Hair dryer', 232: 'Hair spray', 233: 'Hamburger', 234: 'Hammer', 235: 'Hamster', 236: 'Hand dryer', 237: 'Handbag', 238: 'Handgun', 239: 'Harbor seal', 240: 'Harmonica', 241: 'Harp', 242: 'Harpsichord', 243: 'Hat', 244: 'Headphones', 245: 'Heater', 246: 'Hedgehog', 247: 'Helicopter', 248: 'Helmet', 249: 'High heels', 250: 'Hiking equipment', 251: 'Hippopotamus', 252: 'Home appliance', 253: 'Honeycomb', 254: 'Horizontal bar', 255: 'Horse', 256: 'Hot dog', 257: 'House', 258: 'Houseplant', 259: 'Human arm', 260: 'Human beard', 261: 'Human body', 262: 'Human ear', 263: 'Human eye', 264: 'Human face', 265: 'Human foot', 266: 'Human hair', 267: 'Human hand', 268: 'Human head', 269: 'Human leg', 270: 'Human mouth', 271: 'Human nose', 272: 'Humidifier', 273: 'Ice cream', 274: 'Indoor rower', 275: 'Infant bed', 276: 'Insect', 277: 'Invertebrate', 278: 'Ipod', 279: 'Isopod', 280: 'Jacket', 281: 'Jacuzzi', 282: 'Jaguar (Animal)', 283: 'Jeans', 284: 'Jellyfish', 285: 'Jet ski', 286: 'Jug', 287: 'Juice', 288: 'Kangaroo', 289: 'Kettle', 290: 'Kitchen & dining room table', 291: 'Kitchen appliance', 292: 'Kitchen knife', 293: 'Kitchen utensil', 294: 'Kitchenware', 295: 'Kite', 296: 'Knife', 297: 'Koala', 298: 'Ladder', 299: 'Ladle', 300: 'Ladybug', 301: 'Lamp', 302: 'Land vehicle', 303: 'Lantern', 304: 'Laptop', 305: 'Lavender (Plant)', 306: 'Lemon', 307: 'Leopard', 308: 'Light bulb', 309: 'Light switch', 310: 'Lighthouse', 311: 'Lily', 312: 'Limousine', 313: 'Lion', 314: 'Lipstick', 315: 'Lizard', 316: 'Lobster', 317: 'Loveseat', 318: 'Luggage and bags', 319: 'Lynx', 320: 'Magpie', 321: 'Mammal', 322: 'Man', 323: 'Mango', 324: 'Maple', 325: 'Maracas', 326: 
'Marine invertebrates', 327: 'Marine mammal', 328: 'Measuring cup', 329: 'Mechanical fan', 330: 'Medical equipment', 331: 'Microphone', 332: 'Microwave oven', 333: 'Milk', 334: 'Miniskirt', 335: 'Mirror', 336: 'Missile', 337: 'Mixer', 338: 'Mixing bowl', 339: 'Mobile phone', 340: 'Monkey', 341: 'Moths and butterflies', 342: 'Motorcycle', 343: 'Mouse', 344: 'Muffin', 345: 'Mug', 346: 'Mule', 347: 'Mushroom', 348: 'Musical instrument', 349: 'Musical keyboard', 350: 'Nail (Construction)', 351: 'Necklace', 352: 'Nightstand', 353: 'Oboe', 354: 'Office building', 355: 'Office supplies', 356: 'Orange', 357: 'Organ (Musical Instrument)', 358: 'Ostrich', 359: 'Otter', 360: 'Oven', 361: 'Owl', 362: 'Oyster', 363: 'Paddle', 364: 'Palm tree', 365: 'Pancake', 366: 'Panda', 367: 'Paper cutter', 368: 'Paper towel', 369: 'Parachute', 370: 'Parking meter', 371: 'Parrot', 372: 'Pasta', 373: 'Pastry', 374: 'Peach', 375: 'Pear', 376: 'Pen', 377: 'Pencil case', 378: 'Pencil sharpener', 379: 'Penguin', 380: 'Perfume', 381: 'Person', 382: 'Personal care', 383: 'Personal flotation device', 384: 'Piano', 385: 'Picnic basket', 386: 'Picture frame', 387: 'Pig', 388: 'Pillow', 389: 'Pineapple', 390: 'Pitcher (Container)', 391: 'Pizza', 392: 'Pizza cutter', 393: 'Plant', 394: 'Plastic bag', 395: 'Plate', 396: 'Platter', 397: 'Plumbing fixture', 398: 'Polar bear', 399: 'Pomegranate', 
400: 'Popcorn', 401: 'Porch', 402: 'Porcupine', 403: 'Poster', 404: 'Potato', 405: 'Power plugs and sockets', 406: 'Pressure cooker', 407: 'Pretzel', 408: 'Printer', 409: 'Pumpkin', 410: 'Punching bag', 411: 'Rabbit', 412: 'Raccoon', 413: 'Racket', 414: 'Radish', 415: 'Ratchet (Device)', 416: 'Raven', 417: 'Rays and skates', 418: 'Red panda', 419: 'Refrigerator', 420: 'Remote control', 421: 'Reptile', 422: 'Rhinoceros', 423: 'Rifle', 424: 'Ring binder', 425: 'Rocket', 426: 'Roller skates', 427: 'Rose', 428: 'Rugby ball', 429: 'Ruler', 430: 'Salad', 431: 'Salt and pepper shakers', 432: 'Sandal', 433: 'Sandwich', 434: 'Saucer', 435: 'Saxophone', 436: 'Scale', 437: 'Scarf', 438: 'Scissors', 439: 'Scoreboard', 440: 'Scorpion', 441: 'Screwdriver', 442: 'Sculpture', 443: 'Sea lion', 444: 'Sea turtle', 445: 'Seafood', 446: 'Seahorse', 447: 'Seat belt', 448: 'Segway', 449: 'Serving tray', 450: 'Sewing machine', 451: 'Shark', 452: 'Sheep', 453: 'Shelf', 454: 'Shellfish', 455: 'Shirt', 456: 'Shorts', 457: 'Shotgun', 458: 'Shower', 459: 'Shrimp', 460: 'Sink', 461: 'Skateboard', 462: 'Ski', 463: 'Skirt', 464: 'Skull', 465: 'Skunk', 466: 'Skyscraper', 467: 'Slow cooker', 468: 'Snack', 469: 'Snail', 470: 'Snake', 471: 
'Snowboard', 472: 'Snowman', 473: 'Snowmobile', 474: 'Snowplow', 475: 'Soap dispenser', 476: 'Sock', 477: 'Sofa bed', 478: 'Sombrero', 479: 'Sparrow', 480: 'Spatula', 481: 'Spice rack', 482: 'Spider', 483: 'Spoon', 484: 'Sports equipment', 485: 'Sports uniform', 486: 'Squash (Plant)', 487: 'Squid', 488: 'Squirrel', 489: 'Stairs', 490: 'Stapler', 491: 'Starfish', 492: 'Stationary bicycle', 493: 'Stethoscope', 494: 'Stool', 495: 'Stop sign', 496: 'Strawberry', 497: 'Street light', 498: 'Stretcher', 499: 'Studio couch', 500: 'Submarine', 501: 'Submarine sandwich', 502: 'Suit', 503: 'Suitcase', 504: 'Sun hat', 505: 'Sunglasses', 506: 'Surfboard', 507: 'Sushi', 508: 'Swan', 509: 'Swim cap', 510: 'Swimming pool', 511: 'Swimwear', 512: 'Sword', 513: 'Syringe', 514: 'Table', 515: 'Table tennis racket', 516: 'Tablet computer', 517: 'Tableware', 518: 'Taco', 519: 'Tank', 520: 'Tap', 521: 'Tart', 522: 'Taxi', 523: 'Tea', 524: 'Teapot', 525: 'Teddy bear', 526: 'Telephone', 527: 'Television', 528: 'Tennis ball', 529: 'Tennis racket', 530: 'Tent', 531: 'Tiara', 532: 'Tick', 533: 'Tie', 534: 'Tiger', 535: 'Tin can', 536: 'Tire', 537: 'Toaster', 538: 'Toilet', 539: 'Toilet paper', 540: 'Tomato', 541: 'Tool', 542: 'Toothbrush', 543: 'Torch', 544: 'Tortoise', 545: 'Towel', 546: 'Tower', 547: 'Toy', 548: 'Traffic light', 549: 'Traffic sign', 550: 'Train', 551: 'Training bench', 552: 'Treadmill', 553: 'Tree', 554: 'Tree house', 555: 'Tripod', 556: 'Trombone', 557: 'Trousers', 558: 'Truck', 559: 'Trumpet', 560: 'Turkey', 561: 'Turtle', 562: 'Umbrella', 563: 'Unicycle', 564: 'Van', 565: 'Vase', 566: 'Vegetable', 567: 'Vehicle', 568: 'Vehicle registration plate', 569: 'Violin', 570: 'Volleyball (Ball)', 571: 'Waffle', 572: 'Waffle iron', 573: 'Wall clock', 574: 'Wardrobe', 575: 'Washing machine', 576: 'Waste container', 577: 'Watch', 578: 'Watercraft', 579: 'Watermelon', 580: 'Weapon', 581: 'Whale', 582: 'Wheel', 583: 'Wheelchair', 584: 'Whisk', 585: 'Whiteboard', 586: 'Willow', 587: 'Window', 588: 'Window blind', 589: 'Wine', 590: 'Wine glass', 591: 'Wine rack', 592: 'Winter melon', 593: 'Wok', 594: 'Woman', 595: 'Wood-burning stove', 596: 'Woodpecker', 597: 'Worm', 598: 'Wrench', 599: 'Zebra', 600: 'Zucchini'}

model1.class_list = class_list1


models = [model1]

video = cv2.VideoCapture(0)

# Boucle pour lire et traiter chaque frame en temps réel
while True:
    ret, frame = video.read()

    if not ret:
        break

    # Effectuer des traitements sur la frame
    processed_frame = detect(frame, models)

    # Afficher la frame traitée
    cv2.imshow('Processed Frame', processed_frame)

    # Arrêter la boucle si la touche 'q' est enfoncée
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Fermer les fenêtres et libérer les ressources
video.release()
cv2.destroyAllWindows()