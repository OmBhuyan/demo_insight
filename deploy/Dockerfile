# Base image for building
FROM continuumio/miniconda3


RUN apt update && apt install -y gcc && apt clean autoclean && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install openssl && apt-get install ca-certificates
RUN wget --no-check-certificate https://dl.xpdfreader.com/xpdf-tools-linux-4.04.tar.gz && tar -xvf xpdf-tools-linux-4.04.tar.gz && cp xpdf-tools-linux-4.04/bin64/pdftotext /usr/local/bin
WORKDIR /st_app/

# Copy requirements file to container
COPY deploy/env/environment.yml .

# Copy source code to container
COPY . .

# Create Conda Environment and Install the Requirements
RUN conda env create -f /st_app/environment.yml
ENV PATH /opt/conda/envs/ta_nlp/bin:$PATH

WORKDIR /opt/conda/envs/ta_nlp/nltk_data
COPY data/model_files/punkt /opt/conda/envs/ta_nlp/nltk_data
EXPOSE 80

# Run the streamlit app
WORKDIR /st_app/app_files/

CMD ["bash", "-c", "source activate ta_nlp && streamlit run app.py --server.port 80"]


