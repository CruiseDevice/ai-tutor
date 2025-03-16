import {GetObjectCommand, PutObjectCommand, S3Client} from "@aws-sdk/client-s3";
import {getSignedUrl} from "@aws-sdk/s3-request-presigner";

// Initialize S3 client
const s3Client = new S3Client({
  region: process.env.AWS_REGION,
  credentials: {
    accessKeyId: process.env.AWS_ACCESS_KEY_ID || "",
    secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY || "",
  },
});

const bucketName = process.env.S3_PDFBUCKET_NAME || "";

/**
 * Upload a file to S3
 * @param file The file to upload
 * @param key The S3 object key (path)
 * @returns Object with the URL of the uploaded file
 * **/
export async function uploadToS3(file: File | Buffer, key: string, contentType?: string) {
  // if file is a File object, convert to buffer
  let buffer: Buffer;
  let mimeType = contentType;

  if (file instanceof File) {
    buffer = Buffer.from(await file.arrayBuffer());
    mimeType = contentType || file.type;
  } else {
    buffer = file;
  }

  // upload to S3
  const params = {
    Bucket: bucketName,
    Key: key,
    Body: buffer,
    ContentType: mimeType,
  };

  await s3Client.send(new PutObjectCommand(params));

  // Generate a presigned URL for read access (valid for 7 days)
  const getCommand = new GetObjectCommand({
    Bucket: bucketName,
    Key: key
  });

  const url = await getSignedUrl(s3Client, getCommand, {expiresIn: 604800})  // 7 days
  return {
    url,
    key,
  }
}

export async function getSignedS3Url(key: string, expiresIn = 3600) {
  const command = new GetObjectCommand({
    Bucket: bucketName,
    Key: key,
  });

  return getSignedUrl(s3Client, command, {expiresIn});
}