import { useParams, useLocation, Link } from "react-router-dom";
import { useQuery, useMutation } from "@tanstack/react-query";
import { useShowError, Loader, Button } from "src/components";
import { getYachtById } from "src/services/yachts";
import { Stack, Typography } from "@mui/material";
import { EVENTS } from "src/constants/events";
import { StyledImage } from "./styled";
import { addEvent } from "src/services/events";

const YachtDetailsPage = () => {
  const { id } = useParams();
  const location = useLocation();
  const backLinkHref = location.state ?? "/";

  const {
    isPending,
    isError,
    data: yacht,
  } = useQuery({
    queryKey: ["yachts", id],
    queryFn: () => getYachtById(id),
    enabled: !!id,
  });

  useShowError(isError, "There was an error getting yacht details");

  const { mutate: bookEventMutation } = useMutation({
    mutationFn: () =>
      addEvent({
        yachtId: id,
        type: EVENTS.START_BOOKING,
      }),
    onSuccess: () => {
      console.log(
        `${EVENTS.START_BOOKING} event was successfully sent to the server`
      );
      // TODO add logic of adding to the cart
    }
  });

  const onBookNow = () => {
    bookEventMutation();
  };

  return (
    <Stack width="100%" padding={"20px"}>
      {/* <Link to={backLinkHref}>Back to yachts</Link> */}
      {isPending && <Loader />}
      {!isPending && yacht && (
        <Stack gap={4}>
          <Stack direction="row" gap={4} width="100%">
            <Stack width="58%">
              <StyledImage src={yacht.photos[0]} alt="main yacht image" />
            </Stack>
            <Stack width="42%" rowGap={4}>
              <Stack direction="row" gap={4} width="49%">
                <StyledImage src={yacht.photos[1]} alt="yacht image 1" />
                <StyledImage src={yacht.photos[2]} alt="yacht image 1" />
              </Stack>
              <Stack direction="row" gap={4} width="49%">
                <StyledImage src={yacht.photos[3]} alt="yacht image 1" />
                <StyledImage src={yacht.photos[4]} alt="yacht image 1" />
              </Stack>
            </Stack>
          </Stack>
          <Stack direction="row" gap={4} justifyContent={"space-between"}>
            <Typography variant="h3">{yacht.name}</Typography>
            <Button variant="contained" onClick={onBookNow}>
              Book now
            </Button>
          </Stack>
        </Stack>
      )}
    </Stack>
  );
};

export default YachtDetailsPage;
