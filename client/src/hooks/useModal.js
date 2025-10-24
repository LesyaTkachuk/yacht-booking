import { MouseEvent, useState } from "react";

export const useModal = (defaultValue = false) => {
  const [isOpen, setIsOpen] = useState(defaultValue);

  const open = (event) => {
    event?.stopPropagation();
    setIsOpen(true);
  };
  const close = () => setIsOpen(false);
  const toggle = () => setIsOpen((isCurrentOpen) => !isCurrentOpen);

  return { isOpen, open, close, toggle, setIsOpen };
};
