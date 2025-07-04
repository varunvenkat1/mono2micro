from fastapi import FastAPI, Form, HTTPException, UploadFile, File, APIRouter, Depends
from pydantic import BaseModel, HttpUrl
import os
import shutil
import asyncpg
from datetime import datetime
import uuid
import  subprocess
import zipfile
from typing import Optional

from embedding import Embedding
from qwencoderllm import identify_microservices, genrate_microservices, analyse_file
from postgres import get_postgres
from service import *

router = APIRouter()

upload_dir = os.path.join(os.getcwd(), "monolith_apps")
microservice_dir = os.path.join(os.getcwd(), "generated_microservices")
embedding = Embedding()
BATCH_SIZE = 60


@router.get("/get_monolith_info/")
async def get_monolith_info_by_name(monolith_name:str, db_pool: asyncpg.Pool = Depends(get_postgres)):
    try:
        return await get_data_by_name(monolith_name, db_pool)
    except HTTPException as e:
        print(f"Error fetching monolith data: {e}")
        raise e
    except Exception as e:
        print(f"Error fetching monolith data: {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error while fetching monolith data"
        )

@router.get("/get_all/")
async def get_all_monoliths(db_pool: asyncpg.Pool = Depends(get_postgres)):
    try:
        return await get_all(db_pool)
    except Exception as e:
        print(f"Error retrieving monolith data: {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error while retrieving monolith data"
        )


async def upload_files(files: list[UploadFile],
                       db_pool: asyncpg.Pool = Depends(get_postgres)):
    source_code_dir = os.path.join(upload_dir, "test")
    if os.path.exists(source_code_dir):
        shutil.rmtree(source_code_dir)
    os.makedirs(source_code_dir)

    for file in files:
        file_path = os.path.join(source_code_dir, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
    await insert_db_record("test", "test url", db_pool)
    await embed_code(source_code_dir)
    return {"message": "Files uploaded successfully!"}
   
@router.post("/add_monolith/")
async def add_monolith(name:str,
                       language:str,
                       zip_file: Optional[UploadFile] = File(None),
                       git_url:str = "",
                       db_pool: asyncpg.Pool = Depends(get_postgres)):
    name_exist = await check_if_name_already_added(name, db_pool)
    if name_exist:
        raise HTTPException(
            status_code=400, detail="Given name already exist in the database."
        )
    
    if git_url != "":
        url = git_url
        #id = str(uuid.uuid4())
        monolith_path = os.path.join(upload_dir, name)
        if(os.path.exists(monolith_path)):
            shutil.rmtree(monolith_path)
        try:
            print("Cloning repository...")
            subprocess.run(["git", "clone", url, monolith_path], check=True)
            #subprocess.run(["git", "clone", "--branch", name, url, monolith_path], check =True)
            #branch = "main"  # or "master"
            #subprocess.run(["git", "clone", "--branch", branch, url, monolith_path], check=True)
            print("Repository clone successful!")
            await insert_db_record(name, "git | " + url, language, db_pool)
            print("Repository record added to database.")
            return {"message": "Repo downloaded successfully!", "path": monolith_path}
        except HTTPException as e:
            raise e
        except Exception as e:
            raise HTTPException(status_code=500, detail="failed to clone repository.")
    #elif request.files.count() > 0:
    #    try:
    #        await upload_files(request.files, request.name)
    #        print("Files uploaded successful!")
    #        await insert_db_record(request.name, f"{request.files.count()} files uploaded", db_pool)
    #        print("Record added to database.")
    #        return {"message": "Files uploaded successfully!", "path": monolith_path}
    #    except Exception as e:
    #        raise HTTPException(status_code=500, detail="failed to clone repository.")
    else:
        try:
            zip_path = os.path.join(upload_dir, zip_file.filename)
    
            with open(zip_path, "wb") as buffer:
                shutil.copyfileobj(zip_file.file, buffer)

            monolith_path = os.path.join(upload_dir, name)
            # Extract ZIP file
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(monolith_path)

            print({"message": "ZIP file extracted successfully", "extracted_to": monolith_path})

            await insert_db_record(name, f"zip | {zip_file.filename}", language, db_pool)
            print("Repository record added to database.")
            return {"message": "Repo extracted from zip file successfully!", "path": monolith_path}

        except zipfile.BadZipFile:
            raise HTTPException(status_code=400, detail="Invalid ZIP file")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

async def insert_summary(
    
    monolith_name: str,
    monolith_analysis: str,
    db_pool: asyncpg.Pool = Depends(get_postgres),
) -> str:
    query = """
        INSERT INTO monotomicro_summary (monolith_name, monolith_analysis)
        VALUES ($1, $2)
    """
    try:
        async with db_pool.acquire() as conn:
            result = await conn.fetchrow(query, monolith_name, monolith_analysis)
            return "monolith_analysis updated!"
    except Exception as e:
        print(f"Error updating monolith_analysis: {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error during updating monolith_analysis"
        )

async def check_if_name_already_added(name: str, db_pool):
    query = "select monolith_name from monotomicro"
    async with db_pool.acquire() as conn:
        result = await conn.fetch(query)
        names = [record[0] for record in result]
        print (names)
        return name in names

async def insert_db_record(
    monolith_name : str,
    monolith_url : str,
    language: str,
    db_pool: asyncpg.Pool,
) -> str:
    query = """
        INSERT INTO monotomicro (monolith_name, monolith_url, monolith_analysis, microservice_suggestion, date_time, status, language, action)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
    """
    try:
        async with db_pool.acquire() as conn:
            result = await conn.execute(query, monolith_name, monolith_url, "", "", datetime.now(), "uploaded", language, "embedding")
            return "monolith files added!"
    except Exception as e:
        print(f"Error inserting repo data: {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error during inserting monolith data"
        )
"""
@router.post("/add_monolith")
async def add_monolith(
    name: str = Form(...),
    language: str = Form(...),
    git_url: str = Form(""),
    zip_file: Optional[UploadFile] = File(None),
    db_pool: asyncpg.Pool = Depends(get_postgres)
):
    name_exist = await check_if_name_already_added(name, db_pool)
    if name_exist:
        raise HTTPException(
            status_code=400, detail="Given name already exists in the database."
        )

    monolith_path = os.path.join(upload_dir, name)

    if git_url.strip():
        if os.path.exists(monolith_path):
            shutil.rmtree(monolith_path)

        try:
            print("Cloning repository...")
            subprocess.run(
                ["git", "clone", "--branch", "main", git_url, monolith_path],
                check=True
            )
            print("Repository cloned successfully.")

            await insert_db_record(name, f"git | {git_url}", language, db_pool)
            return {"message": "Repo cloned successfully!", "path": monolith_path}
        except subprocess.CalledProcessError as e:
            raise HTTPException(status_code=500, detail="Failed to clone repository.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    elif zip_file is not None:
        try:
            zip_path = os.path.join(upload_dir, zip_file.filename)

            with open(zip_path, "wb") as buffer:
                shutil.copyfileobj(zip_file.file, buffer)

            if os.path.exists(monolith_path):
                shutil.rmtree(monolith_path)

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(monolith_path)

            await insert_db_record(name, f"zip | {zip_file.filename}", language, db_pool)
            return {"message": "ZIP file extracted successfully!", "path": monolith_path}
        except zipfile.BadZipFile:
            raise HTTPException(status_code=400, detail="Invalid ZIP file.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    else:
        raise HTTPException(
            status_code=400, detail="Either a ZIP file or Git URL must be provided."
        )
"""
async def update_monolith_analysis(
    monolith_name: str,
    monolith_analysis: str,
    db_pool: asyncpg.Pool = Depends(get_postgres),
) -> str:
    query = """
        UPDATE monotomicro
        SET monolith_analysis = $1, date_time = $2, status = $3, action = $4 where monolith_name = $5
    """
    try:
        async with db_pool.acquire() as conn:
            result = await conn.fetchrow(query, monolith_analysis, datetime.now(), "embeddings_created", "identify_microservices", monolith_name)
            return "analysis summary inserted!"
    except Exception as e:
        print(f"Error updating monolith_analysis: {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error during updating monolith_analysis"
        )


@router.post("/embedding/")
async def embed_code(monolith_name: str,
                     language: str,
                     db_pool: asyncpg.Pool = Depends(get_postgres)):
    
    await embedding.add_dotnet_codebase_embeddings(monolith_name, os.path.join(upload_dir,monolith_name), language)
    for root, _, files in os.walk(os.path.join(upload_dir,monolith_name)):
            for file in files:
                if file.endswith("." + language):
                    file_path = os.path.join(root, file)
                    with open(file_path, "r", encoding="utf-8") as f:
                        code = f.read()
                        context = embedding.retrieve_similar_code(monolith_name, code)
                        result = await analyse_file(context, code)
                        await insert_summary(monolith_name, result, db_pool)
    return {"message": "Code processed and stored in vector store."}

async def update_microservice_suggestion(
    monolith_name: str,
    microservice_suggestion: str,
    db_pool: asyncpg.Pool = Depends(get_postgres),
) -> str:
    query = """
        UPDATE monotomicro
        SET microservice_suggestion = $1, date_time = $2, status = $3,action = $4 where monolith_name = $5
    """
    try:
        async with db_pool.acquire() as conn:
            result = await conn.fetchrow(query, microservice_suggestion, datetime.now(), "identified_microservices","code_generation", monolith_name)
            return "microservice_suggestion updated!"
    except Exception as e:
        print(f"Error updating microservice_suggestion: {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error during updating microservice_suggestion"
        )
        
@router.get("/get_microservice_suggestion/")
async def get_microservice_suggestion_by_name(name, db_pool: asyncpg.Pool = Depends(get_postgres)):
    query = "select microservice_suggestion from monotomicro where monolith_name = $1"
    try:
        async with db_pool.acquire() as conn:
            result = await conn.fetch(query, name)
            return result
    except Exception as e:
        print(f"Error retrieving monolith data: {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error while retrieving monolith data"
        )


@router.post("/identify_microservices/")
async def generate_microservice_suggestion(monolith_name: str,
                                           db_pool: asyncpg.Pool = Depends(get_postgres)):
    offset = 0
    all_results = []

    try:
        async with db_pool.acquire() as conn:
            while True:
                query = """
                    SELECT monolith_analysis
                    FROM monotomicro_summary
                    WHERE monolith_name = $1
                    ORDER BY id  -- or timestamp or another ordering field
                    LIMIT $2 OFFSET $3
                """
                rows = await conn.fetch(query, monolith_name, BATCH_SIZE, offset)

                if not rows:
                    break  # No more data

                # Combine summaries into one prompt
                combined_summary = "\n".join(f"- {row['monolith_analysis']}" for row in rows)

                # Call LLM-based microservice suggestion
                result = await identify_microservices(combined_summary)
                all_results.append(result)

                offset += BATCH_SIZE

            # Store the final result (you can customize how you aggregate it)
            final_combined_result = "\n---\n".join(all_results)
            await update_microservice_suggestion(monolith_name, final_combined_result, db_pool)

            return {"microservices": all_results}

    except Exception as e:
        print(f"Error processing microservice suggestions: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during microservice identification")


@router.post("/generate_microservices/")
async def genrate_microservices_code(microservice_name: str):
    os.makedirs(microservice_dir, exist_ok = True)

    result = embedding.retrieve_code_for_microservice(microservice_name)

    microservice_code = genrate_microservices(result)

    return microservice_code
        
    service_name = f"microservice_{idx}.py"
    service_path = os.path.join(microservice_dir, service_name)

    with open(service_path, "w", encoding="utf-8") as f:
        f.write(microservice_code)
    
    return {"message": f"Microservices generated in {microservice_dir}"}

@router.post("/save_microservice_suggestion/")
async def save_microservice_suggestion(monolith_name: str, microservice_suggestion: str,
                                           db_pool: asyncpg.Pool = Depends(get_postgres)):
    await update_microservice_suggestion(monolith_name,microservice_suggestion, db_pool)
    return {"microservices": microservice_suggestion}