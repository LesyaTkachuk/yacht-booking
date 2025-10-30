import { Typography, Divider, Stack } from "@mui/material";
import { useMutation } from "@tanstack/react-query";
import { useNavigate, useLocation, generatePath } from "react-router-dom";
import DetailItem from "./DetailItem";
import {
  StyledListItem,
  StyledImageWrapper,
  StyledImage,
  StyledIconButton,
} from "./styled";
import HeartIcon from "src/assets/icons/heart-empty.svg";
import LocationIcon from "src/assets/icons/location.svg";
import CalendarIcon from "src/assets/icons/calendar.svg";
import ArrowsIcon from "src/assets/icons/arrows.svg";
import BedIcon from "src/assets/icons/bed.svg";
import UsersIcon from "src/assets/icons/users.svg";
import { ROUTES } from "src/navigation/routes";
import { EVENTS } from "src/constants/events";
import { addEvent } from "src/services/events";

const YachtCard = ({ yachtDetails }) => {
  const navigate = useNavigate();
  const location = useLocation();

  const { mutate: viewEventMutation } = useMutation({
    mutationFn: () =>
      addEvent({
        yachtId: yachtDetails.id,
        type: EVENTS.VIEW,
      }),
    onSuccess: () => {
      console.log(`${EVENTS.VIEW} event was successfully sent to the server`);
    },
  });

  const onYachtClick = () => {
    viewEventMutation();
    navigate(generatePath(ROUTES.YACHT_DETAILS, { id: yachtDetails.id }), {
      state: location,
    });
  };

  return (
    <StyledListItem onClick={onYachtClick}>
      <StyledImageWrapper>
        <StyledImage src={yachtDetails.photos[0]} alt={yachtDetails.name} />
        <StyledIconButton>
          <HeartIcon />
        </StyledIconButton>
      </StyledImageWrapper>
      <Stack>
        <Typography variant="subtitle2" textAlign={"left"}>
          {yachtDetails.name}
        </Typography>
        <Stack direction={"row"} justifyContent={"space-between"}>
          <Typography>${yachtDetails.summerLowSeasonPrice} / day</Typography>
          <Typography color="secondary.dark">{yachtDetails.type}</Typography>
        </Stack>
        <Divider
          color="secondary.main"
          lineHeight={2}
          style={{ margin: "6px 0" }}
        />
        <Stack direction={"row"}>
          <DetailItem
            icon={CalendarIcon}
            keyTitle="Year"
            value={yachtDetails.year}
          />
          <DetailItem
            icon={ArrowsIcon}
            keyTitle="Length"
            value={yachtDetails.length}
          />
          <DetailItem
            icon={LocationIcon}
            keyTitle="Location"
            value={yachtDetails.baseMarina}
          />
          <DetailItem
            icon={BedIcon}
            keyTitle="Cabins"
            value={yachtDetails.cabins}
          />
          <DetailItem
            icon={UsersIcon}
            keyTitle="Capacity"
            value={yachtDetails.guests}
          />
        </Stack>
      </Stack>
    </StyledListItem>
  );
};

export default YachtCard;
