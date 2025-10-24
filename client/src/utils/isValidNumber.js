const isValidNumber = (value) =>
  !!value && !Number.isNaN(value) && !!(Number(value) >= 0);

export default isValidNumber;
