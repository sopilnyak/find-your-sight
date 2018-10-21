const GoogleImages = require('google-images');
const fs = require('fs');
const path = require('path');
const mkdirp = require('mkdirp');

const numPages = 40;
const targetDir = 'images';
const placesFile = 'places.txt';
const cxId = '';
const apiKey = '';

const client = new GoogleImages(cxId, apiKey);

String.prototype.replaceAll = function(search, replacement) {
    let target = this;
    return target.replace(new RegExp(search, 'g'), replacement);
};

function processImages(images, filename) {
    let imageUrls = [];
    images.forEach(image =>
        imageUrls.push(image.url));
    fs.appendFile(filename, imageUrls.join('\n'), () => {});
}

function processPlace(placeName) {
    if (placeName.length === 0) {
        return;
    }
    let filename = path.join(targetDir, placeName.replaceAll(' ', '-') + '.txt');
    console.log('Saving to ' + filename);
    for (let i = 0; i < numPages; i++) {
        client.search(placeName, {'page': i})
            .then(images => {
                processImages(images, filename);
            });
    }
}

function processFile(inputFile) {
    let places = [];
    let readline = require('readline'),
        instream = fs.createReadStream(inputFile),
        outstream = new (require('stream'))(),
        rl = readline.createInterface(instream, outstream);

    rl.on('line', function (line) {
        if (line !== undefined) {
            places.push(line.toString());
        }
    });

    rl.on('close', function (line) {
        if (line !== undefined) {
            places.push(line.toString());
        }
        places.forEach(place => {
            if (place.length !== 0) {
                processPlace(place.toString())
            }
        });
    });
}

mkdirp(targetDir);
processFile(placesFile);
