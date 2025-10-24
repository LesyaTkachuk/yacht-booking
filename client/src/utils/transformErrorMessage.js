import capitalizeFirstLetter from "./capitalizeFirstLetter";
import isObject from "./isObject";

const transformErrorMessage = (message, level = 0, separator = ", ") => {
  let resultMessage = "";

  if (typeof message === "string") {
    return message;
  }

  if (Array.isArray(message)) {
    return message
      .map((message) => transformErrorMessage(message))
      .join(separator);
  }

  if (isObject(message)) {
    for (const [key, value] of Object.entries(message)) {
      const formattedKey = capitalizeFirstLetter(key);
      const indentation = "- ".repeat(level);

      if (Array.isArray(value)) {
        resultMessage += `${indentation}${formattedKey}: ${value
          .map((message) => transformErrorMessage(message))
          .join(", ")}\n`;
      } else if (isObject(value)) {
        resultMessage += `${indentation}${formattedKey}:\n${transformErrorMessage(
          value,
          level + 1
        )}`;
      } else {
        resultMessage += `${indentation}${formattedKey}: ${transformErrorMessage(
          value
        )}\n`;
      }
    }
  }

  return resultMessage.trim();
};

export default transformErrorMessage;
