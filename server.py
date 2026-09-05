import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.main_stream:app", host="localhost", port=8000, reload=True)