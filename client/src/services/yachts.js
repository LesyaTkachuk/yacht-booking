import axios from "./axiosInstance.js";

export const getYachts = async () =>
  await axios.get("/yachts", { params: { limit: 18 } });

export const getTopBookedYachts = async () =>
  await axios.get("/yachts//top-booked");

export const getRecommendedYachts = async () =>
  await axios.get("/yachts/recommendations");

export const getUserYachts = async () => await axios.get("/yachts/own");

export const getYachtById = async (id) => await axios.get(`/yachts/${id}`);

export const getSimilarYachts = async (id) =>
  await axios.get(`/yachts/${id}/similar`);

export const addYacht = async (yacht) => await axios.post("/yachts", yacht);

export const removeYacht = async (id) => await axios.delete(`/yachts/${id}`);

export const updateYacht = async (id, yacht) =>
  await axios.patch(`/yachts/${id}`, yacht);

export const updateYachtRating = async (id, rating) =>
  await axios.patch(`/yachts/${id}/rating`, { rating });

export const getPresignedUrl = async (name, index, fileType) => {
  const response = await axios.get("/yachts/upload-url", {
    params: { name, index, fileType },
  });
  return response;
};

export const uploadFileToR2 = async (uploadUrl, file) => {
  const response = await fetch(uploadUrl, {
    method: "PUT",
    body: file,
    headers: {
      "Content-Type": file.type,
    },
  });

  if (!response.ok) {
    const errorText = await response.text();
    console.error("R2 Upload Error:", errorText);
    throw new Error(`Failed to upload to R2: ${response.status} ${response.statusText}`);
  }
};

