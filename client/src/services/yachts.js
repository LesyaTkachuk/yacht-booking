import axios from "./axiosInstance.js";

export const getYachts = async () =>
  await axios.get("/yachts", { params: { limit: 18 } });

export const getUserYachts = async () => await axios.get("/yachts/own");

export const getYachtById = async (id) => await axios.get(`/yachts/${id}`);

export const addYacht = async (yacht) => await axios.post("/yachts", yacht);

export const removeYacht = async (id) => await axios.delete(`/yachts/${id}`);

export const updateYacht = async (id, yacht) =>
  await axios.patch(`/yachts/${id}`, yacht);

export const updateYachtRating = async (id, rating) =>
  await axios.patch(`/yachts/${id}/rating`, { rating });
