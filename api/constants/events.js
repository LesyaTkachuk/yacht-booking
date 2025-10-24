export const EVENTS = {
  VIEW: "view",
  WISHLIST: "wishlist",
  CHAT_OWNER: "chatOwner",
  START_BOOKING: "startBooking",
  BOOK: "book",
};

export const EVENT_WEIGHTS = {
  [EVENTS.VIEW]: 2,
  [EVENTS.WISHLIST]: 4,
  [EVENTS.CHAT_OWNER]: 6,
  [EVENTS.START_BOOKING]: 8,
  [EVENTS.BOOK]: 10,
};
