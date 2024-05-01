import os
import warnings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import List

warnings.filterwarnings("ignore", category=DeprecationWarning)

from src.routers import (
    insights_router
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(insights_router.router,tags=["Insights"])

@app.get("/")
async def main():
    content = """
    <body>
    <form action="/upload" enctype="multipart/form-data" method="post">
    <input name="files" type="file" multiple>
    <input type="submit">
    </form>
    </body>
    """
    return {"message": "Please go to cg-hack-insight-services.azurewebsites.net/docs to test the API"}


