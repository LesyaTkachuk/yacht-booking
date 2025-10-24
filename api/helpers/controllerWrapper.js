const controllerWrapper = (controller) => {
  const func = async (req, res, next) => {
    try {
      await controller(req, res, next);
    } catch (error) {
      if (error.name === "SequelizeValidationError") {
        error.status = 400;
      }
      if (error.name === "SequelizeUniqueConstraintError") {
        error.status = 409;
      }
      next(error);
    }
  };
  return func;
};

export default controllerWrapper;
