import os

source_dir = 'tmp'

for filename in os.listdir(source_dir):
    extension = filename.split('.')[-1]
    orig_name = os.path.join(source_dir, filename)

    if extension == 'aspx' or extension == 'ashx':
        os.remove(orig_name)
        print("Removed {}".format(orig_name))
        continue

    possible_extensions = {'jpg', 'jpeg', 'png', 'bmp', 'gif'}

    if extension.lower() not in possible_extensions:
        target_name = os.path.join(source_dir, filename[:-3] + 'jpg')
        os.rename(orig_name, target_name)
        print("Renamed {} to {}".format(orig_name, target_name))

for filename in os.listdir(source_dir):
    orig_name = os.path.join(source_dir, filename)
    if int(os.stat(orig_name).st_size) < 250:
        os.remove(orig_name)
        print("Removed {}".format(orig_name))

names_orig = ["Basilica of the Sagrada Familia",
              "Gothic Quarter",
              "The Font Mágica Fountain",
              "Picasso Museum",
              "Camp Nou Stadium",
              "Park Guell By Gaudi",
              "Barcelona Cathedral",
              "Barcelona Zoo",
              "Poble Espanyol",
              "La Rambla Market",
              "La Boqueria Food Market",
              "Montjuic Hill",
              "Casa Batllo",
              "Montjuic Castle",
              "Casa Mila",
              "Aquarium Barcelona",
              "Montserrat",
              "Palau de la Musica Orfeo Catalana",
              "Barceloneta Beach",
              "Placa d’Espanya",
              "FC Barcelona Museum",
              "Santa Maria del Mar",
              "Parc de la Ciutadella",
              "Tibidabo Amusement Park",
              "Arc de Triomf",
              "Anella Olimpica",
              "Mirador - Collserola Tower",
              "Barcelona Bosc Urbà",
              "Columns of the Temple of Augustus"]

names = ["Basilica of the Sagrada Familia",
         "Gothic Quarter",
         "The Font Magica Fountain",
         "Picasso Museum",
         "Camp Nou Stadium",
         "Park Guell By Gaudi",
         "Barcelona Cathedral",
         "Barcelona Zoo",
         "Poble Espanyol",
         "La Rambla Market",
         "La Boqueria Food Market",
         "Montjuic Hill",
         "Casa Batllo",
         "Montjuic Castle",
         "Casa Mila",
         "Aquarium Barcelona",
         "Montserrat",
         "Palau de la Musica Orfeo Catalana",
         "Barceloneta Beach",
         "Placa dEspanya",
         "FC Barcelona Museum",
         "Santa Maria del Mar",
         "Parc de la Ciutadella",
         "Tibidabo Amusement Park",
         "Arc de Triomf",
         "Anella Olimpica",
         "Mirador Collserola Tower",
         "Barcelona Bosc Urba",
         "Columns of the Temple of Augustus"]

i = 361
for filename in os.listdir(source_dir):
    orig_name = os.path.join(source_dir, filename)

    id = filename.split('-')[-1].split('.')[0]
    if 'FC Barcelona Museum'.replace(' ', '-') in orig_name:
        target_name = orig_name.replace('FC Barcelona Museum'.replace(' ', '-'),
                                        'Camp Nou Stadium'.replace(' ', '-')).replace(id, str(i))
        os.rename(orig_name, target_name)
        print("Renamed {} to {}".format(orig_name, target_name))
        i += 1

    for i, name in enumerate(names_orig):
        replaced_name = name.replace(' ', '-')
        target_name = os.path.join(source_dir, filename.replace(replaced_name, names[i].replace(' ', '-')))
        if orig_name != target_name:
            os.rename(orig_name, target_name)
            print("Renamed {} to {}".format(orig_name, target_name))
