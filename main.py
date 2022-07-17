from pydoc import describe
from tokenize import String
from fastapi import FastAPI, Path, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from create_bb import get_bboxes

from paddleocr import PaddleOCR,draw_ocr
# Paddleocr supports Chinese, English, French, German, Korean and Japanese.
# You can set the parameter `lang` as `ch`, `en`, `french`, `german`, `korean`, `japan`
# to switch the language model in order.
ocr = PaddleOCR(use_angle_cls=True, lang='en') # need to run only once to download and load model into memory

from pdf2image import convert_from_path
import os
from typing import List, Text
import pymongo
from pymongo import MongoClient
from typing import Optional
import uuid
from torch_snippets import Glob
import hashlib

templates = Jinja2Templates(directory="html")


# client = pymongo.MongoClient("mongodb+srv://abhay:a1s2d3f4g5h6j7k8@ocrtestcluster.9h0zrm2.mongodb.net/?retryWrites=true&w=majority")
client = pymongo.MongoClient("mongodb://localhost:27017")
db=client["ocrtestdb"]
collection=db["ocrtestcollection"]

app=FastAPI()


@app.get("/")
async def main():
    content = """
                <body>
                <h2>a) prepare_bounding_box</h2>
                <form action="/prepare_bounding_box/" enctype="multipart/form-data" method="post">
                <input name="img" type="file">
                <input type="submit">
                </form>

                <h2>b) prepare_raw_transcription_from_bb</h2>
                <form action="/prepare_raw_transcription_from_bb/" enctype="multipart/form-data" method="post">
                <input name="img" type="file">
                <input type="submit">
                </form>

                <h2>c) prepare_raw_transcription = step(a)+step(b)</h2>
                <form action="/prepare_raw_transcription/" enctype="multipart/form-data" method="post">
                <input name="img" type="file">
                <input type="submit">
                </form>

                <h2>c) prepare_raw_transcription from paddleocr>
                <form action="/paddleocr/" enctype="multipart/form-data" method="post">
                <input name="img" type="file">
                <input type="submit">
                </form>

                <h2>d) Search:
                <form action="/search/" >
                
                <input type="text" id="query" name="query" value="Himachal Pradesh"><br><br>
                <input type="submit" value="Submit">
                </form>
                """

    return HTMLResponse(content=content)

def convert_and_save_pdf2image(files):

    imgs = []
    for file in files:
        if (file.filename[-4:]==".pdf"):
            pages = convert_from_path(file, 500)
            for idx, page in enumerate(pages):
                save_path = 'inputs/{}_{}.jpg'.format(file.filename[:-4], idx)
                imgs.append(save_path)
                page.save(save_path, 'JPEG')
        else:
            save_path = 'inputs/{}'.format(file.filename)
            imgs.append(save_path)

            with open(save_path, "wb+") as file_object:
                file_object.write(file.file.read())
    
    return imgs


from get_raw_text import raw_text

@app.post("/prepare_raw_transcription")
async def prepare_raw_transcription(imgs: List[UploadFile] = File(...)):
    imgs = convert_and_save_pdf2image(imgs)

    for img in imgs:
        file_location = img
        text=raw_text(file_location)
        id=uuid.uuid4().hex
        collection.insert_one({"_id":id, "transcription":text})

    return{"Saved all images"}

@app.post("/prepare_bounding_box")
async def prepare_bounding_box(imgs: List[UploadFile] = File(...)):
    imgs = convert_and_save_pdf2image(imgs)

    for img in imgs:
        file_location = img
        bb_path=get_bboxes(file_location)

    return {"Message": "Bounding box saved at 'bbs' folder."}


@app.post("/prepare_raw_transcription_from_bb")
async def prepare_raw_transcription_from_bb(imgs: List[UploadFile] = File(...)):
    imgs = convert_and_save_pdf2image(imgs)
    for img in imgs:

        file_location = img
        filename = file_location.split("/")[-1]
        bbs_file=".".join(filename.split(".")[:-1])
        bbs_location = "bbs/{}".format(bbs_file+".bbs")
        if not (os.path.exists(bbs_location)):
            print("BBox doesnt seems to exist for this iage. Please create BBox first.")

        text=raw_text(file_location, bbs_location)
        id = hashlib.sha256(text.encode('utf-8')).hexdigest()

        if (collection.find_one({"_id":id})):
            print("Transcription already exists with id: {}".format(id))
        else:
            collection.insert_one({"_id":id, "transcription":text})

    return{"Done!"}

@app.post("/paddleocr")
async def paddleocr(imgs: List[UploadFile] = File(...)):
    imgs = convert_and_save_pdf2image(imgs)

    for img in imgs:
        file_location = img
        result = ocr.ocr(file_location, cls=True)
        text=""
        for line in result:
            text+=" "+line[-1][0]
        id = hashlib.sha256(text.encode('utf-8')).hexdigest()

        if (collection.find_one({"_id":id})):
            print("Transcription already exists with id: {}".format(id))
        else:
            collection.insert_one({"_id":id, "transcription":text})

    return{"Done!"}

@app.get("/search")
async def search( request: Request, query):
    results = collection.find({'$text':{'$search':query}})  

    return templates.TemplateResponse("loop_list.html", {
        'request': request,
        "results": results,
    })

@app.get("/get_transcription")
async def get_transcription( request: Request, id):
    results = collection.find_one({'_id':id}) 

    return templates.TemplateResponse('print_transcription.html', {
        'request': request,
        "results": results,
    })