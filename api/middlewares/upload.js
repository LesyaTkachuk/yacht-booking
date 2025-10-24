import multer from "multer";
import path from "path";
import HttpError from "../helpers/HttpError.js";

const destination = path.resolve("temp");

const storage = multer.diskStorage({
  destination,
  filename: (_, file, cb) => {
    const uniquePrefix = Date.now() + "-" + Math.round(Math.random() * 1e9);
    cb(null, uniquePrefix + "-" + file.originalname);
  },
});

const limits = {
  fileSize: 5 * 1024 * 1024, // 5MB
};

const fileFilter = (req, file, cb) => {
  const extension = file.originalname.split(".").pop();
  if (extension === "exe") {
    return cb(new HttpError(400, "File type not supported"), false);
  }
  cb(null, true);
};

const upload = multer({ storage, limits, fileFilter });

export default upload;
