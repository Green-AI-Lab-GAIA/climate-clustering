import os
from io import BytesIO
from os import path

import pandas as pd
from office365.runtime.auth.user_credential import UserCredential
from office365.sharepoint.client_context import ClientContext
from office365.sharepoint.files.file import File
from tqdm import tqdm

def print_success(*args, **kwargs):
    print("\033[92m", *args, "\033[0m", **kwargs)


class SharePointConnection:
    def __init__(
        self,
        username: str,
        password: str,
        sharepoint_site,
        sharepoint_site_name,
        sharepoint_doc,
    ):
        """
        Class to connect to SharePoint and read/write files

        Parameters
        ----------
        username : str
            Email used to login in SharePoint
        password : str
            Password used to login in SharePoint
        sharepoint_site : str
            SharePoint site URL
        sharepoint_site_name : str
            SharePoint site name
        sharepoint_doc : str
            SharePoint shared documents folder name e.g. "Documentos Compartilhados", "Shared Documents"
        """
        self.username = username
        self.password = password
        self.sharepoint_site = sharepoint_site
        self.sharepoint_site_name = sharepoint_site_name
        self.sharepoint_doc = sharepoint_doc
        # verify credentials
        self.conn = self._auth()
        try:
            self.conn.web.get().execute_query()
            print("\033[92m Conexão com o SharePoint realizada com sucesso \033[0m")
        except ValueError:
            raise ValueError("Credenciais inválidas")
        finally:
            # print("Conexão com o SharePoint realizada com sucesso")
            # print green
            pass

    def _auth(self):
        conn = ClientContext(self.sharepoint_site).with_credentials(
            UserCredential(self.username, self.password)
        )
        return conn

    def _get_files_list(self, folder_name):
        conn = self._auth()
        target_folder_url = f"{self.sharepoint_doc}/{folder_name}"
        root_folder = conn.web.get_folder_by_server_relative_url(target_folder_url)
        root_folder.expand(["Files", "Folders"]).get().execute_query()
        return root_folder.files

    def get_folder_list(self, folder_name: str) -> list:
        conn = self._auth()
        target_folder_url = f"{self.sharepoint_doc}/{folder_name}"
        root_folder = conn.web.get_folder_by_server_relative_url(target_folder_url)
        root_folder.expand(["Folders"]).get().execute_query()
        folder_names = list()
        for folder in root_folder.folders:
            folder_path = folder.properties["Name"]
            folder_names.append(folder_path)
        return folder_names

    def get_file_list(self, folder_name: str) -> list:
        conn = self._auth()
        root_folder = conn.web.get_folder_by_server_relative_url(
            f"{self.sharepoint_doc}/{folder_name}"
        )
        files = root_folder.expand(["Files"]).get().execute_query()
        file_names = list()
        for file in files.files:
            file_path = file.properties["ServerRelativeUrl"]
            file_name = file_path.split("/")[-1]
            file_names.append(file_name)
        return file_names

    def show_items_folder(self, folder_name: str) -> list:
        folder_names = self.get_folder_list(folder_name)
        file_names = self.get_file_list(folder_name)
        print(folder_name + "/" + "\n")
        for folder in folder_names:
            print("\t" + folder + "/")
        for file in file_names:
            print("\t" + file)

    def read_parquet(self, file_path: str, **kwargs):
        buffer = BytesIO()
        conn = self._auth()
        file_url = (
            f"/sites/{self.sharepoint_site_name}/{self.sharepoint_doc}/{file_path}"
        )
        file = File.open_binary(conn, file_url)
        buffer.write(file.content)
        df = pd.read_parquet(buffer, **kwargs)
        return df

    def read_csv(self, file_path: str, **kwargs):
        conn = self._auth()
        file_url = (
            f"/sites/{self.sharepoint_site_name}/{self.sharepoint_doc}/{file_path}"
        )

        file = File.open_binary(conn, file_url)
        df = pd.read_csv(BytesIO(file.content), **kwargs)
        return df

    def write_parquet(self, df: pd.DataFrame, file_path: str, **kwargs):
        conn = self._auth()

        buffer = BytesIO()
        df.to_parquet(buffer, **kwargs)

        folder_name = "/".join(file_path.split("/")[:-1])
        file_name = file_path.split("/")[-1]
        target_folder_url = f"{self.sharepoint_doc}/{folder_name}"
        root_folder = conn.web.get_folder_by_server_relative_url(target_folder_url)

        root_folder.upload_file(file_name, buffer.getvalue()).execute_query()

        print_success("File uploaded successfully")

    def write_csv(self, df: pd.DataFrame, file_path: str, **kwargs):
        conn = self._auth()

        buffer = BytesIO()
        df.to_csv(buffer, index=False, mode="wb", encoding="utf-8", **kwargs)

        folder_name = "/".join(file_path.split("/")[:-1])
        file_name = file_path.split("/")[-1]
        target_folder_url = f"{self.sharepoint_doc}/{folder_name}"

        root_folder = conn.web.get_folder_by_server_relative_url(target_folder_url)
        try:
            root_folder.upload_file(file_name, buffer.getvalue()).execute_query()
        except:
            raise ValueError("O caminho do arquivo não existe")

        print_success("File uploaded successfully")
    
    def upload_file(self, local_path:str, remote_folder_path:str):
        
        file_name = local_path.split("/")[-1]
        
        target_folder_url = f"{self.sharepoint_doc}/{remote_folder_path}"
        
        root_folder = self.conn.web.get_folder_by_server_relative_url(target_folder_url)
        
        with open(local_path, "rb") as file_input:
            file_content = file_input.read()
            
        root_folder.upload_file(file_name, file_content).execute_query()
    
    
    def download_file(self, remote_path:str, local_path:str):
        
        conn = self._auth()
        
        file_name = remote_path.split("/")[-1]
        
        file_url = f"/sites/{self.sharepoint_site_name}/{self.sharepoint_doc}/{remote_path}"
        
        file = File.open_binary(conn, file_url)
        
        if path.isdir(local_path):
            output_file_path = path.join(local_path, file_name)
        
        else:
            output_file_path = local_path
            
        with open(output_file_path, "wb") as file_output:
            file_output.write(file.content)
        
        print_success("File downloaded successfully")
    
    
    def create_remote_folder(self, remote_folder:str, new_folder_name:str):
        
        conn = self._auth()
                
        target_folder_url = f"{self.sharepoint_doc}/{remote_folder}"
        
        root_folder = conn.web.get_folder_by_server_relative_url(target_folder_url)
            
        # check if folder already exists
        for folder in root_folder.folders.get().execute_query():
            if folder.properties["Name"] == new_folder_name:
                return
        
        root_folder.add(new_folder_name).execute_query()
        print_success(f"Folder {new_folder_name} created successfully")
    


    def upload_folder(self, local_folder:str, remote_folder:str, progress_bar=True):
        
        parent_local_folder, local_folder_name = path.split(local_folder)
    
        self.create_remote_folder(remote_folder, local_folder_name)
        
        # create all subfolders in remote_folder
        for root, dirs, files in os.walk(local_folder):
            for dir in dirs:
                remote_subfolder = path.relpath(root, parent_local_folder)
                self.create_remote_folder(remote_subfolder, dir)
        
        
        # upload files
        total_files = sum([len(files) for _, _, files in os.walk(local_folder)])
        pbar = tqdm(total=total_files, desc="Uploading files", disable=not progress_bar)
        for root, dirs, files in os.walk(local_folder):
            for file in files:
                local_path = path.join(root, file)
                remote_path = path.join(remote_folder, path.relpath(root, parent_local_folder))
                self.upload_file(local_path, remote_path)
                pbar.update(1)
        
        pbar.close()
        print_success("Folder uploaded successfully")
    

