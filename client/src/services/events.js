import axios from "./axiosInstance.js";

export const addEvent = async (event) => await axios.post("/events", event);
