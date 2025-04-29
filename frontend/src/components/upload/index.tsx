import React, { useCallback, useState } from "react";
import Button from "@mui/material/Button";
import { useDropzone, FileWithPath } from "react-dropzone";

const Upload: React.FC = () => {
  const [imageSrc, setImageSrc] = useState<string | ArrayBuffer | null>(null);
  const [file, setFile] = useState<File | null> (null); 

  const onDrop = useCallback((acceptedFiles: FileWithPath[]) => {
    const file  = acceptedFiles[0];
    setFile(file);
    setImageSrc(URL.createObjectURL(file));
  }, []);

  const { getRootProps, getInputProps} = useDropzone({
    accept: {
      "image/*": ["png", "jpeg", "jpg"],
    },
    maxFiles: 1,
    onDrop,
  });

  const onClick = async () => {
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    fetch('http://127.0.0.1:5000/predict', {
      method: "POST",
      body: formData,
    });
  };
  return(
    <>
      {imageSrc && (
        <img src={imageSrc} />
      )}
      <div {...getRootProps()}>
        <input {...getInputProps()} />
        <p>Drag 'n' drop some files here, or click to select files</p>
      </div>

      <Button disabled={!imageSrc} onClick={onClick}>Send</Button>
    </>
  );
};

export default Upload;