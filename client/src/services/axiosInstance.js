import axios from "axios";

const baseURL = import.meta.env.VITE_BASE_API_URL ?? "/api";

export const HEADER_MFD = { "Content-Type": "multipart/form-data" };

const axiosInstance = axios.create({
  baseURL,
  headers: {
    "Content-Type": "application/json",
  },
});

axiosInstance.interceptors.request.use(
  async (config) => {
    const accessToken = localStorage.getItem("token");
    if (accessToken) {
      config.headers.Authorization = `Bearer ${accessToken}`;
    } else {
      delete config.headers.Authorization;
    }
    return config;
  },
  (error) => Promise.reject(error?.response || error)
);

axiosInstance.interceptors.response.use(
  (response) => response.data,
  (error) => Promise.reject(error?.response || error)
);

export default axiosInstance;
