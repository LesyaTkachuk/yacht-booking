import axios, { HEADER_MFD } from './axiosInstance.js';

export const registerUser = async userData => await axios.post('/auth/register', userData);

export const loginUser = async credentials => await axios.post('/auth/login', credentials);

export const logoutUser = async () => await axios.post('/auth/logout');

export const getCurrentUser = async () => await axios.get('/auth/current');

export const updateRole=async()=>await axios.patch('/auth/roles');

export const updateAvatar = async formData => await axios.patch('/auth/avatar', formData, { headers: HEADER_MFD });