export const MISSING_AUTHORIZATION_HEADER = "Missing Authorization header";
export const MISSING_BEARER = "Missing Bearer";
export const USER_NOT_AUTHORIZED = "User not authorized";
export const USER_NOT_FOUND = "User not found";
export const USER_NOT_VERIFIED = "User is not verified";
export const USER_NOT_FOUND_OR_VERIFIED = "User not found or already verified";
export const USER_VERIFIED = "User already verified";
export const USER_ALREADY_EXISTS = "User already exists";
export const INVALID_CREDENTIALS = "Email or password is wrong";
export const FILE_UPLOAD_ERROR = "File upload error";
export const getUserExistMessage = (email) =>
  `User with email ${email} already exists`;
export const getYachtNotFoundMessage = (id) => `Yacht with id=${id} not found`;

export const EVENT_NOT_FOUND = "Event not found";
export const ACCESS_DENIED = "Access denied";
export const PERMISSION_DENIED =
  "You don't have permission to perform this action";
